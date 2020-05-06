"""
Fuse images from different sensors using the principles of multivariate alteration detection.

@author: Nick Leach
"""

import os
import sys
import rasterio
import rasterio.mask
import rasterio.features
import rasterio.warp
from rasterio.enums import Resampling
import affine
import fiona
import numpy as np
import numpy.ma as ma
from geopandas import GeoDataFrame
import pandas as pd
from shapely.geometry import shape
import datetime
import gdal
import warnings
import time
import multiprocessing as mp
import scipy.ndimage
import geojson
import json
import geopandas as gpd
from shapely.geometry import Polygon
import logging
from shapely.geometry import LinearRing
# weird and janky, but works until these libraries are combined
try:
    from despike import despike
    import segment_fitter as sf
except ModuleNotFoundError:
    sys.path.insert(1, r"C:\Users\nleach\PycharmProjects\untitled")
    from despike import despike
    import segment_fitter as sf

log = logging.getLogger(__name__)


def qa_mask(qa_band):
    """
    Create a binary mask based on which pixels are flipped in the QA band.
    By default, excludes cirrus, clouds, adjacent cloud, cloud shadow, and snow/ice.
    :param qa_band: (8int 1D np array)
    """
    clear = [0, 32, 64, 96, 128, 160, 192, 224]  # good pixels
    ones = np.ones(np.shape(qa_band))
    msk = np.isin(qa_band, clear)  # create binary array, where the clear pixels are marked as True
    out = ma.masked_array(ones, ~msk)  # mask everything except the clear pixels
    return out


def read_hdf_to_arr(hdf_path, bands, datatype=np.int16):
    """
    Functionalizing the process of reading HDF files into arrays
    read a single band out of the hdf and load it into a numpy array
    """
    if os.path.isfile(hdf_path):
        if type(bands) in (list, tuple, np.ndarray):  # if multiple bands are passed
            arr = []
            src = gdal.Open(hdf_path)
            for band in bands:
                band_ds = gdal.Open(src.GetSubDatasets()[band][0], gdal.GA_ReadOnly)
                band_array = band_ds.ReadAsArray().astype(datatype)
                arr.append(band_array)
            del src
            arr = np.array(arr)
            return arr
        else:
            src = gdal.Open(hdf_path)
            band_ds = gdal.Open(src.GetSubDatasets()[bands][0], gdal.GA_ReadOnly)
            band_array = band_ds.ReadAsArray().astype(datatype)
            return band_array
    else:
        warnings.warn("{} does not exist".format(hdf_path))
        return


def get_hdf_crs(hdf_path):
    if os.path.isfile(hdf_path):
        if ".S30" in os.path.split(hdf_path)[1]:
            product = "S30"
        elif ".L30" in os.path.split(hdf_path)[1]:
            product = "L30"
        else:
            log.exception("Can't identify product")
        hdr_path = hdf_path + ".hdr"
        print(hdr_path)
        if not os.path.exists(hdr_path):
            log.exception("Couldn't find hdr file: {}".format(hdr_path))
        else:
            with open(hdr_path, 'r') as f:
                guts = f.read()
                pieces = guts.split('\n')
                crs = pieces[11].split('=')
                crs = crs[1][2:-1]
                transform = pieces[10].split('=')
                transform_list = transform[1].strip()[1:].split(',')
                ulx = transform_list[3].strip()
                uly = transform_list[4].strip()
                resx = transform_list[5].strip()  # resolution
                resy = transform_list[6].strip()
                transform = [ulx, uly, resx, resy]
    else:
        log.exception("That file does not exist")
    return crs, transform


def convert_hdf_to_tif(hdf_path, outdir, bands, apply_qa_mask=False, nodataval=-1000):
    """
    :param hdf_path:
    :param outdir:
    :param bands: list-like
    :return:
    """
    if apply_qa_mask:
        qa_arr = read_hdf_to_arr(hdf_path, 13)
        qa_mask_arr = qa_mask(qa_arr)
        negatives = np.zeros_like(qa_mask_arr, dtype=bool, subok=False)  # get the dimensions of the image from the QA mask
        arr = np.zeros((len(bands), np.shape(qa_mask_arr)[0], np.shape(qa_mask_arr)[1]))  # pre-allocate
        # apply QA mask and remove all negative values
        for count, b in enumerate(bands):
            tmp_band = read_hdf_to_arr(hdf_path, b)
            tmp = ma.masked_array(tmp_band, qa_mask_arr.mask)
            arr[count, :, :] = ma.filled(tmp, nodataval)  # fill mask using nodata value. add to 'arr' to create a 4-band image
            # find all the negative pixels
            negatives = negatives + (tmp_band < 0)  # True where there are negative values. False elsewhere.
        arr = ma.masked_array(arr, mask=np.broadcast_to(negatives[np.newaxis, :, :], arr.shape))
        arr = ma.filled(arr, nodataval)

    else:
        arr = []
        negatives = []
        for count, b in enumerate(bands):
            tmp_band = read_hdf_to_arr(hdf_path, b)
            if count == 0:
                # need to get dimensions
                arr = np.zeros((len(bands), tmp_band.shape[0], tmp_band.shape[1]))
                negatives = np.zeros_like(arr, dtype=bool, subok=False)
            arr[count, :, :] = ma.filled(tmp_band, nodataval)
            negatives = negatives + (tmp_band < 0)
        arr = ma.masked_array(arr, negatives)
        arr = ma.filled(arr, nodataval)

    with rasterio.open(hdf_path) as src:
        kwds = src.profile

    kwds['nodata'] = nodataval
    kwds['driver'] = 'GTiff'
    kwds['dtype'] = rasterio.int16
    kwds['width'] = arr.shape[2]
    kwds['height'] = arr.shape[1]
    kwds['count'] = arr.shape[0]
    crs, transform = get_hdf_crs(hdf_path)
    kwds['crs'] = crs
    kwds['transform'] = rasterio.transform.from_origin(float(transform[0]), float(transform[1]),
                                                       float(transform[2]), float(transform[3]))
    outname = os.path.splitext(os.path.split(hdf_path)[1])[0] + ".tif"
    outpath = os.path.join(outdir, outname)
    with rasterio.open(outpath, 'w', **kwds) as dst:
        dst.write(np.array(arr).astype(rasterio.int16))

    return outpath


def clip_to_shapefile(raster, shapefile, outname="clipped_raster.tif", outdir=None):
    """
    Clip the input raster to the given shapefile.

    :param raster: (str) path to raster to clip
    :param shapefile: (str) path to shapefile with features to use for clipping
    :param outname: (str) name of the output raster
    :param outdir: (str) if given, save the output this folder
    :return:
    """
    if outdir:
        # if outdir is specified, save the clipped raster there
        outpath = os.path.join(outdir, outname)
    else:
        # otherwise, save to the same folder as the input raster
        outpath = os.path.join(os.path.split(raster)[0], outname)
    # load in the features from shapefile
    with fiona.open(shapefile, 'r') as src:
        features = [feature['geometry'] for feature in src]
    # create clipped raster data and transform
    with rasterio.open(raster, 'r') as src:
        out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
        out_meta = src.meta.copy()
    # update metadata with new height, width, and transform
    out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    # save to outpath
    with rasterio.open(outpath, 'w', **out_meta) as dst:
        dst.write(out_image)
    return outpath


def make_shapefile_from_raster(raster, outname="vectorized.shp", outdir=None):
    """
    Generate a shapefile with a single feature outlining the extent of the input raster.
    NB: this only works well if there are no no-data areas within the main raster extent (i.e. only around the edges)

    :param raster: (str) path to raster to vectorize
    :param outname: (str) name of the generated shapefile
    :param outdir: (str) if given, save the output to this folder
    :return:
    """
    if outdir:
        # if outdir is specified, save the clipped raster there
        outpath = os.path.join(outdir, outname)
    else:
        # otherwise, save to the same folder as the input raster
        outpath = os.path.join(os.path.split(raster)[0], outname)
    d = dict()
    d['val'] = []
    geometry = []
    with rasterio.open(raster, 'r') as src:
        msk = src.dataset_mask()  # load in the mask (data==255, no-data==0)
        for shp, val in rasterio.features.shapes(source=msk, transform=src.transform):  # use mask for geometry
            if val == 255:  # value of unmasked areas
                d['val'].append(val)
                geometry.append(shape(shp))
        raster_crs = src.crs
    df = pd.DataFrame(data=d)
    geo_df = GeoDataFrame(df, crs={'init': raster_crs['init']}, geometry=geometry)
    geo_df['area'] = geo_df.area
    geo_df.to_file(outpath, driver="ESRI Shapefile")
    return outpath


def make_intersection_poly(img1_path, img2_path, outname="intersect_poly.shp", outdir=None):
    """

    :param img1_path: raster
    :param img2_path: raster
    :param outname:
    :param outdir:
    :return:
    """
    if not outdir:
        outdir = os.path.split(img1_path)[0]
    img1_shapefile = make_shapefile_from_raster(img1_path, os.path.split(img1_path)[1][:-4]+".shp")
    img2_shapefile = make_shapefile_from_raster(img2_path, os.path.split(img2_path)[1][:-4]+".shp")

    polygon1 = [shape(feature['geometry']) for feature in fiona.open(img1_shapefile)][0]
    polygon2 = [shape(feature['geometry']) for feature in fiona.open(img2_shapefile)][0]
    intersect_poly = polygon1.intersection(polygon2)
    d = dict()
    d['val'] = [255]  # the fill value for data in rasterio masks
    geometry = [intersect_poly]
    # might end up with geometry[0] being a Geometry Collection.
    # If that's the case, go through it and delete all non-polygons
    g = None
    try:
        for g in geometry[0]:  # loop through the geometry collection, if that's what it is
            if g.geom_type != "Polygon":  # delete anything that isn't a polygon
                del g
        geometry[0] = g  # grab the first remaining item from the geometry collection, set that to geometry[0]
    except TypeError:  # if it's not a geometry collection, pass
        pass
    with rasterio.open(img1_path, 'r') as src:
        raster_crs = src.crs
    df = pd.DataFrame(data=d)
    geo_df = GeoDataFrame(df, crs={"init": raster_crs["init"]}, geometry=geometry)
    geo_df["area"] = geo_df.area
    intersect_out = os.path.join(outdir, outname)
    geo_df.to_file(intersect_out, driver="ESRI Shapefile")
    return intersect_out


def get_ncp(mad_img, change_method="chi2", thresh=100, save_image=False, outname="ncp_img.tif", out_path=None):
    """
    Calculate the no-change probability of each pixel in an input MAD image.
    Assumes that the sum of the squared standardized MAD variates follows a chi-squared distribution.
    May not be a good assumption.
    :param mad_img: (string) path to the MAD image
    :param change_method: (str) change detection method. Default to chi2.
        chi2: uses a chi-squared test, assuming a chi2 distribution with (bands-1) degrees of freedom
        threshold: if chosen, the output will be change/no-change rather than a probability. thresh is min for change.
    :param thresh: (numeric) if change_method=="threshold", use this value as minimum band value for change
    :param save_image: (bool) whether or not to save the NCP values as a single band GeoTiff
    :param outname: (str) filename of the NCP image
    :param out_path: (str) output destination of the NCP image
    :return:
    """
    # read in the MAD image's chisquared band (last band) and ravel it
    with rasterio.open(mad_img, 'r') as src:
        # get info about the image (rows, cols, bands)
        profile = src.meta
        bands = profile["count"]
        chisqr = src.read(bands)  # even if not using chi2 method, use this name to refer to sum of sqrd MAD variates
    chisqr_flat = chisqr.ravel()

    ncp_reshape = None
    running = True  # force user to choose a valid change_method
    while running:
        if change_method == "chi2":
            running = False  # finish while loop if change_method is chi2
            from scipy.stats import chi2
            # generate no-change probability for each pixel (location on normalized chi2 cdf - basically alpha value)
            ncp = 1 - chi2.cdf(chisqr_flat, bands-1)
            # ncp = no-change probability, so low number means low prob of being unchanged.
            # i.e. high numbers are invariant
            ncp_reshape = np.reshape(ncp, np.shape(chisqr))
            # if requested, save a new image with the ncp for each image
        elif change_method == "threshold":
            running = False
            changed_pix = chisqr_flat > thresh  # if a pixel exceeds thresh, mark as changed (True)
            ncp_reshape = np.reshape(changed_pix, np.shape(chisqr))
        else:
            change_method = input("Invalid change method. Options are 'chi2' and 'threshold': ")

    if save_image:
        # if path is not specified, give it a silly name and save it with the input MAD image
        if not out_path:
            out_dir = os.path.split(mad_img)[0]
            out_path = os.path.join(out_dir, outname)
        profile["count"] = 1
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(ncp_reshape)

    return ncp_reshape  # return the reshaped array


def parse_hls_filepath(filepath):
    """
    Creates a dictionary with some basic info about the input HLS image based on its filename.
    Assumes default filenames.
    :param filepath: (string)
    :return: (dict)
    """
    filename = os.path.split(filepath)[1]
    file_info = dict()
    file_info["extension"] = filename[-4:]
    file_info["product"] = filename[4:7]
    file_info["year"] = int(filename[-16:-12])
    file_info["doy"] = int(filename[-12:-9])
    datetime_object = datetime.date(file_info["year"], 1, 1) + datetime.timedelta(file_info["doy"] - 1)
    file_info["month"] = datetime_object.month
    file_info["dom"] = datetime_object.day
    file_info["datetime"] = datetime_object
    return file_info


def parse_ps_filepath(filepath):
    filename = os.path.split(filepath)[1]
    file_info = dict()
    file_info["extension"] = filename[-4:]
    file_info["year"] = int(filename[0:4])
    file_info["month"] = int(filename[4:6])
    file_info["dom"] = int(filename[6:8])
    file_info["product"] = "ps_sr"
    datetime_object = datetime.date(file_info["year"], file_info["month"], file_info["dom"])
    file_info["doy"] = datetime_object.timetuple().tm_yday
    file_info["datetime"] = datetime_object
    return file_info


def collect_image_info(directory, image_type="hls"):
    """
    Get basic info about every image in directory.
    Return a dictionary. Keys are filepaths, values are other dictionaries with info about the filepath
    :param directory: (string) file extension, year, day of year, month, day of month, datetime object
    :param image_type: (string) either "hls" or "ps".
    :return: (dict)
    """
    image_info_all = {}
    all_filepaths = []
    for root_dir, dirnames, files in os.walk(directory):
        for f in files:
            all_filepaths.append(os.path.join(root_dir, f))
    while (image_type.lower() != "hls") and (image_type.lower() != "ps"):
        image_type = input("Which image type? hls or ps? ")
    if image_type.lower() == "hls":
        for f in [f for f in all_filepaths if f.endswith(".tif")]:
            image_info_all[f] = parse_hls_filepath(f)
    else:
        for f in [f for f in all_filepaths if f.endswith("merge.tif") or f.endswith("FINAL.tif")]:
            image_info_all[f] = parse_ps_filepath(f)
    return image_info_all


def find_closest_date(image_date, parsed_files):
    """
    Return index of list_of_dates with the date closest to image_date.
    Generally, this does not distinguish between dates that earlier in the year or later in the year than image_date.
    However, if there are two dates equally near image_date, preference will be given to the earlier year.
    :param image_date: (datetime obj)
    :param parsed_files: (dictionary) output from collect_image_info. Dictionary of dictionaries.
    :return:
    """
    date_differences = []
    for ref in parsed_files:
        date_differences.append([(image_date - parsed_files[ref]["datetime"]).days, ref])  # positive when image_date is later than test date
        # this method means that the preference is to normalize to images earlier in the year
    idx_min = np.argmin([abs(d[0]) for d in date_differences])  # index of the date in list which is closest to image_date
    date_differences = np.asarray(date_differences)
    nearest_image = date_differences[idx_min][1]
    return nearest_image


def buffer_and_downsample(target_image, reference_image, outname="reprojected.tif", outdir=None, dst_nodata=0.0,
                          dst_dtype='uint16', resample=Resampling.cubic):
    """
    Resample target_image to match resolution of reference_image. If reference_image is larger than target_image, then
    buffer target_image with no-data values so that it has the same width and height as reference_image.
    :param target_image: (str) file path of the image with the data to be resampled
    :param reference_image: (str) file path of the image with the desired resolution and extent
    :param dst_nodata: (numeric) no-data value for the output image
    :param dst_dtype: (str) data type for the output image
    :param outname: (str) name of output file
    :param outdir: (str) directory to save output image. If not given, image will be saved in working directory
    :param resample: rasterio resampling method, e.g. Resample.bilinear, Resample.cubic
    :return:
    """
    outpath = os.path.join(outdir, outname)

    with rasterio.open(target_image, 'r') as src:
        tgt_data = src.read()
        tgt_meta = src.profile
    with rasterio.open(reference_image, 'r') as src:
        ref_meta = src.profile
    dst_res = ref_meta['transform'][0]   # destination resolution in meters. Assumes square pixels
    left = ref_meta['transform'][2]  # western extent of reference_image
    top = ref_meta['transform'][5]  # northern extent of reference_image
    new_transform = affine.Affine(dst_res, 0., left, 0., -1.*dst_res, top)  # transformation for new image to be saved
    dst_raster = np.zeros((ref_meta['count'], ref_meta['height'], ref_meta['width']), dtype=dst_dtype)  # store data
    # reproject the data from target_image into the array dst_raster using the new transformation and the resolution of
    # the reference_image
    rasterio.warp.reproject(tgt_data, dst_raster, src_transform=tgt_meta['transform'], dst_transform=new_transform,
                            src_crs=tgt_meta['crs'], dst_crs=ref_meta['crs'], dst_nodata=dst_nodata,
                            src_nodata=tgt_meta['nodata'], resampling=resample)
    # update the meta for the output file
    dst_profile = ref_meta
    dst_profile['transform'] = new_transform
    dst_profile['nodata'] = dst_nodata
    dst_profile['dtype'] = dst_dtype
    with rasterio.open(outpath, 'w', **dst_profile) as dst:
        dst.write(dst_raster)
    return outpath


def build_time_series(hls_dir, ps_dir=None, start_year=2016, nodata_val=-1000, verbose=True):
    """
    Loads in all the HLS arrays in hls_dir. If ps_dir is given, then PS data is also loaded.
    For each pixel location, the NDVI is calculated and corresponding dates extracted.
    If a pixel is masked/no-data, it is marked as False in 'keepers'
    :param hls_dir: (str) directory with HLS images. May contain subdirectories
    :param ps_dir: (str) direcotry with PS images, each ending with "FINAL.TIF". May contain subdirectories
    :param start_year: (int) Year of first image
    :param nodata_val: (numeric) no-data value in images. NB: currently requires PS and HLS images to have same no-data
    :param verbose: (bool) if True, prints off each time a row is finished
    :return all_pixel_data: tuple with (ndvi, keepers, is_hls)
    """
    # use a timer!
    start = time.time()
    hls_fileinfo = collect_image_info(hls_dir)  # collect info on all HLS images
    if ps_dir:
        ps_fileinfo = collect_image_info(ps_dir, image_type="ps")
    else:
        ps_fileinfo = None  # just to meet PEP guidelines

    # get all dates, counting from 2016-01-01 by default
    series_dates = []
    is_hls = []  # sort of janky, but we can use this to keep track of which images came from HLS or PS
    for img in hls_fileinfo:
        series_dates.append(hls_fileinfo[img]['doy'] + (hls_fileinfo[img]['year']-start_year)*365)
        is_hls.append(True)  # place an identifier in is_hls for each added date
    if ps_dir:
        ps_dates = []
        for img in ps_fileinfo:
            ps_dates.append(ps_fileinfo[img]['doy'] + (ps_fileinfo[img]['year']-start_year)*365)
            is_hls.append(False)  # identify these dates as not HLS
        series_dates = series_dates + ps_dates
    sorted_dates = sorted(series_dates)  # want a sorted version since the dates may initially be grouped by sensor
    is_hls = np.array([x for y, x in sorted(zip(series_dates, is_hls))])  # sorting is_hls based on dates

    # load in all the HLS arrays
    hls_arrays = []
    for dirpath, dirnames, filenames in os.walk(hls_dir):
        for img in [f for f in filenames if f.endswith(".tif")]:
            img_path = os.path.join(dirpath, img)
            with rasterio.open(img_path) as src:
                hls_arrays.append(src.read())
    # load in all the normalized downsampled PS arrays, if provided
    if ps_dir:
        ps_arrays = []
        for dirpath, dirnames, filenames in os.walk(ps_dir):
            for img in [f for f in filenames if f.endswith("FINAL.tif")]:
                img_path = os.path.join(dirpath, img)
                with rasterio.open(img_path) as src:
                    data = src.read()
                    # replace 0.0 values since they cause nan though division errors
                    data = np.where(data != 0.0, data, -1000)
                    ps_arrays.append(data)
        img_arrays = hls_arrays + ps_arrays
    else:
        img_arrays = hls_arrays

    # get dimensions for the HLS images (should all have identical dimensions; PS images should have same too)
    bands = img_arrays[0].shape[0]
    rows = img_arrays[0].shape[1]
    cols = img_arrays[0].shape[2]

    all_pixel_data = []
    for row in range(rows):
        row_data = []
        for col in range(cols):
            series = []
            for band in range(bands):
                # read in the corresponding series (one pixel through all time and bands)
                series.append([arr[band][row][col] for arr in img_arrays])
            ndvi = np.array((np.array(series[3]) - np.array(series[2])) / (
                        np.array(series[3]) + np.array(series[2])))  # ndvi of array
            # fix (i.e. effectively mask) values where one band was masked but others were not
            ndvi = np.where(ndvi < -1.0, -0.0, ndvi)
            ndvi = np.where(ndvi > 1.0, -0.0, ndvi)
            del series
            # sort the ndvi series based on dates (remember, dates are originally unordered - L30 then S30)
            ndvi = np.array([x for y, x in sorted(zip(series_dates, ndvi))])
            # sort the HLS identifiers to match the dates
            # remove dates where the pixel has been masked/saturated
            keepers = []
            for pix in ndvi:
                if pix == 0.0:  # use 0.0 since NDVI will also be precisely 0.0 when there is no-data in all bands
                    keepers.append(False)
                else:
                    keepers.append(True)
            keepers = np.array(keepers)
            # add the ndvi series for this position to a list of arrays
            row_data.append((ndvi, keepers))
        all_pixel_data.append(row_data)  # this will store it in rows... is that what we want? or better to 'unzip'?
        if verbose:
            if row % 10 == 0:
                print("Finished row " + str(row) + " after " + str(int(time.time()-start)) + " seconds. ")
    end = time.time()
    print("Total time elapsed: " + str(int(end-start)) + " seconds.")
    return all_pixel_data, np.array(sorted_dates), is_hls


def calc_despike_and_segs(ndvi, keepers, sorted_dates, despike_thresh=0.05, max_segs=10, seg_thresh=0.05, is_hls=None,
                          include_ps=True, row_num=None, col_num=None, verbose=True):
    """
    Given a time series for a single (pixel) location, calculate the despiked values and segment values.
    Requires the dates associated with each image and a boolean array stating which values to keep (True) and to ignore.
    If the input values come from a mix of HLS and PS, this can also use only the HLS values.
    :param ndvi:
    :param keepers: (nparray bool)
    :param sorted_dates:
    :param despike_thresh:
    :param max_segs:
    :param seg_thresh:
    :param is_hls: (nparray bool)
    :param include_ps:
    :param row_num: (int) current row for printing updates. only relevant if verbose=True
    :param col_num: (int) current column for printing updates. only relevant if verbose=True
    :param verbose: (bool) print some updates about rows/cols that are having issues or completing
    :return:
    """
    # note that this is writing out lists instead of tuples. which is better??
    despike_and_segs = []
    # if no is_hls array is given, generate one that is all True
    if is_hls is calc_despike_and_segs.__defaults__[3]:
        is_hls = np.full(ndvi.shape, True)
    if not include_ps:
        valid_images = is_hls & keepers
    else:
        valid_images = keepers
    # Need at least 4 valid images in order to fit segments
    if len(valid_images[valid_images > 0]) > 3:
        try:
            despiked = despike(ndvi[valid_images], despike_thresh)[1]
            segs = sf.seg_fit(despiked, max_segs, seg_thresh, np.array(sorted_dates)[valid_images])
        except (IndexError, ValueError, ZeroDivisionError):
            if verbose:
                print("Issue at row " + str(row_num) + " and column " + str(col_num))
            despiked = np.zeros_like(ndvi)
            segs = np.zeros_like(ndvi)
    else:
        if verbose and (col_num != 0):
            print("Not enough valid data at row " + str(row_num) + " and column " + str(col_num))
        despiked = np.zeros_like(ndvi)
        segs = np.zeros_like(ndvi)
    return despiked, segs


def do_all_despike_and_segs(time_series, sorted_dates, despike_thresh, max_segs, seg_thresh, is_hls, ps_included=True,
                            verbose=True):
    """
    For every pixel in the time series (output from build_time_series()), fit despike and fit segments.
    :param time_series: (list) the output from build_time_series()
    :param sorted_dates: (array) all image dates, arranged chronologically
    :param despike_thresh:
    :param max_segs:
    :param seg_thresh:
    :param is_hls: (ndarray) boolean array signifying whether each image is from the HLS dataset
    :param ps_included:
    :param verbose:
    :return: (list) despiked_wo_ps, segs_w_ps, [despiked_w_ps, segs_w_ps]
    """
    # take ndvi_all and keepers_all straight out of build_time_series
    # ndvi_all (and keepers_all and is_hls_all) come in as a list with 'rows' items, each with 'cols' items
    print("==============================")
    print("Despiking and fitting segments")
    print("==============================")
    start = time.time()
    all_pixel_data = []
    for i, row in enumerate(time_series):
        row_data = []
        for j, col in enumerate(row):
            ndvi = col[0]
            keepers = col[1]
            # if there are no ps images, then is_hls should just be an array where every value is True
            despike_wo_ps, segs_wo_ps = calc_despike_and_segs(ndvi, keepers, sorted_dates, despike_thresh, max_segs,
                                                              seg_thresh, is_hls, include_ps=False,
                                                              row_num=i, col_num=j)
            if ps_included:
                despike_w_ps, segs_w_ps = calc_despike_and_segs(ndvi, keepers, sorted_dates, despike_thresh,
                                                                max_segs, seg_thresh, is_hls, include_ps=True,
                                                                row_num=i, col_num=j)
                row_data.append([despike_wo_ps, segs_wo_ps, despike_w_ps, segs_w_ps])
            else:
                row_data.append([despike_wo_ps, segs_wo_ps])
        if verbose:
            if i % 10 == 0:
                print("Despiked and seg-fitted row " + str(i) + " after " + str(int(time.time()-start)) + " secs.")
        all_pixel_data.append(row_data)
    return all_pixel_data


# function that will actually apply the calculation to each chunk
def process_chunk(chunk, c_index, sorted_dates, despike_thresh, max_segs, seg_thresh, is_hls, ps_included=True):
    """

    :param chunk: (nparray) pixels to be processed with dimensions (num_pixels, 2, num_dates)
    :param c_index: (int) index to track which chunk is being processed. Needed to reassemble components correctly.
    :param sorted_dates: (arr)
    :param despike_thresh: (float)
    :param max_segs: (float)
    :param seg_thresh: (float)
    :param is_hls: (arr, bool)
    :param ps_included: (bool)
    :return:
    """
    # c_index is used for tracking which portion of the image this chunk belongs to [0 to n_proc]
    chunk_out = []
    for pix in chunk:  # still looping instead of fully vectorizing, but at least now its in parallel
        ndvi = pix[0]
        keepers = pix[1].astype('bool')
        despike_wo_ps, segs_wo_ps = calc_despike_and_segs(ndvi, keepers, sorted_dates, despike_thresh, max_segs,
                                                          seg_thresh, is_hls, include_ps=False, verbose=False)
        if ps_included:
            despike_w_ps, segs_w_ps = calc_despike_and_segs(ndvi, keepers, sorted_dates, despike_thresh, max_segs,
                                                            seg_thresh, is_hls, include_ps=True, verbose=False)
            chunk_out.append([despike_wo_ps, segs_wo_ps, despike_w_ps, segs_w_ps])
        else:
            chunk_out.append([despike_wo_ps, segs_wo_ps])
    print("Finished chunk " + str(c_index))
    return [chunk_out, c_index]


# TODO: figure out if this can be run safely from within a function, or if it needs to be within a '__main__' protocol
def do_all_despike_and_segs_mp(time_series, sorted_dates, despike_thresh, max_segs, seg_thresh, is_hls,
                               ps_included=True, n_proc=None):
    """
    For every pixel in the time series (output from build_time_series()), fit despike and fit segments.
    Fancy multiprocessing version.
    :param time_series:
    :param sorted_dates:
    :param despike_thresh:
    :param max_segs:
    :param seg_thresh:
    :param is_hls:
    :param ps_included:
    :param n_proc:
    :return:
    """
    # take ndvi_all and keepers_all straight out of build_time_series
    # ndvi_all (and keepers_all and is_hls_all) come in as a list with 'rows' items, each with 'cols' items

    # flatten the time series so we just have pixels instead of rows/cols.
    # len(time_series) is the number of rows (e.g. 1466)
    # each item is the number of columns (e.g. 1958)
    time_series = np.array(time_series)  # (rows, cols, ndvi/keepers, dates).
    rows, cols, two, dates = time_series.shape # Unpack to (pixels, ndvi/keepers, dates).
    time_series = time_series.reshape((rows*cols, two, dates))  # put things in a nice flattened array for now
    # set up the multiprocessing framework
    if not n_proc:  # default value
        n_proc = int(mp.cpu_count() / 2)  # number of processes. use half our cores
    chunksize = int(time_series.shape[0] / n_proc)  # size of each chunk to send to each CPU
    # lay out the chunks that we send off for processing
    proc_chunks = []
    for i_proc in range(n_proc):
        chunk_start = i_proc * chunksize
        # include the remainder for the last process since the pixels probably don't divide into the cores evenly
        chunk_end = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None
        proc_chunks.append(time_series[chunk_start:chunk_end, :, :])  # NOT a deep copy - just points to the array
    assert sum(map(len, proc_chunks)) == rows*cols  # make sure we got all the data into proc_chunks
    # need to define a function that processes a chunk. somehow need to track the index of each output so that the
    # partial results can be combined from the individual processes later

    print("==============================")
    print("Despiking and fitting segments")
    print("==============================")
    start = time.time()
    with mp.Pool(processes=n_proc) as pool:
        # each proc_results item (probably) is (data, index)
        proc_results = [pool.apply_async(process_chunk, args=(chunk, c_index, sorted_dates, despike_thresh, max_segs,
                                                              seg_thresh, is_hls))
                        for c_index, chunk in enumerate(proc_chunks)]
        result_chunks = [result.get() for result in proc_results]
    # sort the results based on the index
    results = []
    # I think this loop will order them?? Could probably do some zip/unzip to do this instead
    for index in range(n_proc):
        for result in result_chunks:
            if result[1] == index:
                results.append(result[0])
    # so now results is one big list, presumably with the following 'dimensions': (rows*cols, 4, dates).
    # convert from list to array with correct dimensions
    final_results = np.array(results)
    final_results = np.reshape(final_results, (rows, cols, 4, dates))
    end = time.time()
    print("Total time: " + str((end-start)/3600) + " hours")
    return final_results


def calculate_indices(filepath, outdir=None):
    """
    Generates a 5-band image containing 5 different vegetation indices. In order: NDVI, NGRVI, ARVI, VARIG, GLI_b
    :param filepath: (string) path to input image
    :param outdir: (string) path to output directory. If none, the input directory will be used.
    :return:
    """
    if not outdir:
        outdir = os.path.split(filepath)[0]
    with rasterio.open(filepath, 'r') as src:
        # get the import meta info
        msk = src.dataset_mask()
        meta = src.profile
        # read bands
        blue = np.array(src.read(1), dtype='float32')
        green = np.array(src.read(2), dtype='float32')
        red = np.array(src.read(3), dtype='float32')
        nir = np.array(src.read(4), dtype='float32')
        # multiply by masks to remove any nodata values
        # also mask any values less than -1.0 or greater than +1.0
        ndvi = ma.masked_outside(((nir - red) / (nir + red)) * msk / 255., -1.0, 1.0)
        ngrvi = ma.masked_outside(((green - red) / (green + red)) * msk / 255., -1.0, 1.0)
        arvi = ma.masked_outside(((nir - 2*red + blue) / (nir + 2*red - blue)) * msk / 255., -1.0, 1.0)
        varig = ma.masked_outside(((green - red) / (green + red - blue)) * msk / 255., -1.0, 1.0)
        # actually makes more sense to invert the version of this index that was used by Goodbody et al 2018
        gli_b = ma.masked_outside((-1.0*(2*blue - red - green) / (2*blue + red + green)) * msk / 255., -1.0, 1.0) # inv.
    # create new filename
    filepath_out = os.path.join(outdir, os.path.split(filepath)[1][:-4] + "_VIs.tif")
    # update the meta
    meta['count'] = 5
    meta['dtype'] = 'float32'
    # save image
    with rasterio.open(filepath_out, 'w', **meta) as dst:
        new_msk = msk * (~ndvi.mask | ~ngrvi.mask | ~arvi.mask | ~varig.mask | ~gli_b.mask)
        dst.write_mask(new_msk)
        dst.write(ma.asarray(ndvi, dtype='float32'), 1)
        dst.write(ma.asarray(ngrvi, dtype='float32'), 2)
        dst.write(ma.asarray(arvi, dtype='float32'), 3)
        dst.write(ma.asarray(varig, dtype='float32'), 4)
        dst.write(ma.asarray(gli_b, dtype='float32'), 5)
    return filepath_out


def breakpoint_extraction(segs, dates, return_slopes=False):
    """
    Takes in one segs array and the corresponding dates. Approximates the slope at each point. Uses slope to look for
    breakpoints in the segments.
    Returns the dates of all breakpoints. If there are no breakpoints, returns a length-1 array with 0.
    If the length of segs array is less than 3, returns same result as no breakpoints.
    :param segs: (np array) output from segment fitter
    :param dates: (np array) dates corresponding to each point in segs.
    :param return_slopes: (bool) if True, return the approximate slope at each point in addition to the breakpoints
    :return:
    """
    if len(segs >= 3) and (np.sum(segs) > 0):
        slope = (segs[1:]-segs[:-1])/(dates[1:]-dates[:-1])
        # add correction for when delta_segs == 0 and delta_dates == 0
        fail_count = 0
        while np.isnan(slope).any() and (fail_count < 3):
            fail_count += 1
            try:
                slope = np.where(~np.isnan(slope), slope, np.insert(slope[1:], -1, 0))  # use value from one point earlier
            except IndexError:
                # if that throws an index error, just go back and use the regular slope with nans.
                pass
        derivative = slope[1:]-slope[:-1]  # approximate the slope
        # make an array where dates representing breakpoints are marked as True and all other points are marked as False
        # same length as segs array
        breakpoints = abs(derivative) > 1e-10  # if derivative of segs is significantly greater than 0, then mark change
        breakpoints = np.insert(breakpoints, 0, False)  # buffer with a False value at the beginning
        breakpoints = np.append(breakpoints, False)  # Buffer with a False value at the end too
        if len(dates[breakpoints]) > 0:  # if there are any dates that are breakpoints...
            if return_slopes:  # return the slope at each point if requested. Otherwise, only return breakpoint dates
                return dates[breakpoints], slope  # if there are breakpoints, return them in an array
            else:
                return dates[breakpoints]
        else:  # if no breakpoints
            if return_slopes:
                return np.array([0]), np.array([0])  # return array with 0 for both breakpoints and slope
            else:
                return np.array([0])
    else:
        if return_slopes:
            return np.array([0]), np.array([0])  # return two of these if slopes were requested but unavailable
        else:
            return np.array([0])


def delta_vis(vi_img1, vi_img2, outname="vi_delta.tif", outdir=None):
    """
    Take the difference between two images in their area of overlap. Save the delta image to disk.
    Designed with vegetation indices in mind, but should work fine with other images as well.
    :param vi_img1: (str) filepath to VI image 1. MUST BE THE EARLIER DATE.
    :param vi_img2: (str) filepath to VI image 2. MUST BE THE LATER DATE.
    :param outname:
    :param outdir:
    :return:
    """
    if not outdir:
        outdir = os.path.split(vi_img1)[0]
    # make shapefile of the intersection and clip both images to the area of intersection
    intersection_poly = make_intersection_poly(vi_img1, vi_img2, outname="vi_intersection.shp", outdir=outdir)
    # TODO: do we really need to save these to the disk? Should probably just load them into memory to save write time.
    clipped_date1 = clip_to_shapefile(vi_img1, intersection_poly, outname="vi_date1_clipped.tif", outdir=outdir)
    clipped_date2 = clip_to_shapefile(vi_img2, intersection_poly, outname="vi_date2_clipped.tif", outdir=outdir)
    # load in the clipped images to calculate delta
    with rasterio.open(clipped_date1, 'r') as src:
        data1 = src.read()
        meta1 = src.meta
    with rasterio.open(clipped_date2, 'r') as src:
        data2 = src.read()
        meta2 = src.meta
    delta = data2 - data1  # difference the VIs. Positive means increase in VIs. Negative means decrease.

    # write the results out
    outpath = os.path.join(outdir, outname)
    with rasterio.open(outpath, 'w', **meta1) as dst:
        dst.write(delta)

    return outpath


def change_from_vis_and_mad(vi_path1, vi_path2, mad_img_path, thresh=75, votes=3,
                            decrease_thresh=0, outname="change.tif", outdir=None):
    """
    Areas that experienced change are 1. Elsewhere is 0.
    :param vi_path1: str) filepath to VI image 1. MUST BE THE EARLIER DATE.
    :param vi_path2: (str) filepath to VI image 2. MUST BE THE LATER DATE.
    :param mad_img_path:
    :param thresh:
    :param votes:
    :param decrease_thresh:
    :param outname:
    :param outdir:
    :return:
    """
    if not outdir:
        outdir = os.path.split(mad_img_path)[0]
    delta_vi_path = delta_vis(vi_path1, vi_path2, outdir=outdir)
    mad_change_arr = get_ncp(mad_img_path, change_method="threshold", thresh=thresh, save_image=False)

    with rasterio.open(delta_vi_path, 'r') as src:  # read the delta_vi image created at the top of the function
        delta_vi_arr = src.read()
        # flip GLI_b (band 5) so that positive is associated with higher vegetative health
        delta_vi_arr[-1, :, :] = delta_vi_arr[-1, :, :] * -1.0
        meta_out = src.meta

    vi_change = np.array(delta_vi_arr < decrease_thresh, dtype='uint8')  # anywhere that the VI has decreased is True
    # need to collect "votes" from the vegetation indices
    vi_change_votes = np.sum(vi_change, axis=0)
    # only count change where number of decreased VIs >= votes
    combined_change = mad_change_arr * (vi_change_votes >= votes)

    # save image
    meta_out['count'] = 1  # binary change
    meta_out['dtype'] = 'uint16'  # smallest datatype that we can write to geotiff... I think
    combined_change = combined_change.astype('uint16')
    with rasterio.open(os.path.join(outdir, outname), 'w', **meta_out) as dst:
        dst.write_band(1, combined_change)
    return os.path.join(outdir, outname)


def combine_change_images(temporal_path, spatial_path, outname, outdir=None, include_ps=True):
    """
    Combine change images from breakpoint temporal method with PS-to-PS normalization
    :param temporal_path: (str) path
    :param spatial_path: (str) path
    :param outname:
    :param outdir:
    :param include_ps: (bool) if True, use the HLS+PS from the temporal info. If False, use only the HLS data.
    :return:
    """
    if not outdir:
        outdir = os.path.split(temporal_path)[0]
    with rasterio.open(temporal_path, 'r') as src:
        temporal_meta = src.profile
        if include_ps:
            temporal_arr = src.read(2)  # this is the HLS + PS band!
        else:
            temporal_arr = src.read(1)  # this is the HLS only band!
    with rasterio.open(spatial_path, 'r') as src:
        spatial_meta = src.profile
        spatial_arr = src.read(1)
    dst_res = min(spatial_meta['transform'][0], temporal_meta['transform'][0])
    print("target resolution: " + str(dst_res))

    # figure out the maximum extent (largest of top and right(left?), smallest of bottom and left(right?))
    dst_left = min(spatial_meta['transform'][2], temporal_meta['transform'][2])
    dst_top = max(spatial_meta['transform'][5], temporal_meta['transform'][5])

    right_s = spatial_meta['transform'][2] + spatial_meta['width'] * spatial_meta['transform'][0]
    right_t = temporal_meta['transform'][2] + temporal_meta['width'] * temporal_meta['transform'][0]
    bottom_s = spatial_meta['transform'][5] - spatial_meta['height'] * spatial_meta['transform'][0]
    bottom_t = temporal_meta['transform'][5] - temporal_meta['height'] * temporal_meta['transform'][0]
    dst_right = max(right_s, right_t)
    dst_bottom = min(bottom_s, bottom_t)

    dst_transform = affine.Affine(dst_res, 0.0, dst_left, 0.0, -1. * dst_res, dst_top)

    # next, need to finish building the metadata
    dst_width = int((dst_right - dst_left) / dst_res)
    dst_height = int(-1 * (dst_bottom - dst_top) / dst_res)
    print("dst width: " + str(dst_width))
    print("dst height: " + str(dst_height))

    dst_meta = spatial_meta.copy()
    dst_meta['width'] = dst_width
    dst_meta['height'] = dst_height
    dst_meta['transform'] = dst_transform

    # now that we have the transform and metadata built, need to resample the images using rasterio.reproject

    # building empty raster to store them in
    spatial_resampled = np.zeros((dst_height, dst_width), dtype=dst_meta['dtype'])
    temporal_resampled = np.zeros((dst_height, dst_width), dtype=dst_meta['dtype'])

    # resample into the new empty arrays
    rasterio.warp.reproject(spatial_arr, spatial_resampled, src_transform=spatial_meta['transform'],
                            dst_transform=dst_transform,
                            src_crs=spatial_meta['crs'], dst_crs=dst_meta['crs'], src_nodata=spatial_meta['nodata'],
                            dst_nodata=spatial_meta['nodata'], resampling=Resampling.nearest)

    rasterio.warp.reproject(temporal_arr, temporal_resampled, src_transform=temporal_meta['transform'],
                            dst_transform=dst_transform,
                            src_crs=temporal_meta['crs'], dst_crs=dst_meta['crs'], src_nodata=temporal_meta['nodata'],
                            dst_nodata=dst_meta['nodata'], resampling=Resampling.nearest)
    assert spatial_resampled.shape == temporal_resampled.shape

    # take the pixels that are marked as having change in either figure and use that as a mask for the temporal change
    mask = np.logical_and(spatial_resampled > 0, (temporal_resampled > 0))  # 1 where valid (mult. by 255 for dataset mask)
    temporal_resampled = temporal_resampled * mask

    dst_meta['nodata'] = 0
    # write out a version of the resampled temporal breaks using this as a mask
    with rasterio.open(os.path.join(outdir, outname), 'w', **dst_meta) as dst:
        dst.write_mask(mask.astype('uint8') * 255)  # 255 indicates valid regions
        dst.write_band(1, temporal_resampled)

    return os.path.join(outdir, outname)


def clean_noisy_raster(raster_path, sieve_size=1, erode_size=3, outname="cleaned.tif", outdir=None):
    """
    Clean up a noisy change/no-change raster by removing small features.
    Output is binary change/no-change GeoTiff.
    :param raster_path: (str) raster to clean. Should be a change raster - binary or other small dtype.
    :param sieve_size:
    :param erode_size:
    :param outname:
    :param outdir:
    :return:
    """
    if not outdir:
        outdir = os.path.split(raster_path)[0]

    with rasterio.open(raster_path, 'r') as src:
        meta = src.meta
        msk = src.dataset_mask()
        original_arr = src.read()

    sieved_msk = rasterio.features.sieve(msk, sieve_size, connectivity=8)  # sieve to remove small features

    original_arr_bi = np.where(original_arr <= 0, 0, 1)  # convert to binary change/no-change

    eroded_img = scipy.ndimage.binary_erosion(original_arr_bi[0, :, :], iterations=erode_size)  # erode by erode_size
    reconstruct_img = scipy.ndimage.binary_propagation(eroded_img,
                                                       mask=original_arr_bi[0, :, :])  # propagate until change stops
    tmp = np.logical_not(reconstruct_img)
    eroded_tmp = scipy.ndimage.binary_erosion(tmp, iterations=erode_size)
    filtered_array = np.logical_not(scipy.ndimage.binary_propagation(eroded_tmp, mask=tmp))

    final_out = filtered_array[np.newaxis, :, :].astype('int16')
    # take original_arr and combine it with filtered_array
    final_out = np.where(final_out == 1, original_arr, 0).astype('int16')
    meta['nodata'] = 0
    meta['dtype'] = 'int16'

    with rasterio.open(os.path.join(outdir, outname), 'w', **meta) as dst:
        dst.write(final_out)
        dst.write_mask(sieved_msk)

    return os.path.join(outdir, outname)


def spatial_language(change_img, buffer_size=250, save_metrics=True, outname_metrics="metrics.json",
                     outname="vectorized_change.shp", outdir=None):
    """
    Apply spatial language to create a vector with change, islands, and matrix.
    Save a shapefile with change information.
    Optionally also save a dictionary with some calculated spatial fire metrics.
    :param change_img: (path) raster, preferably binary change/no-change (True/False) and clipped to AOI
    :param buffer_size: (int) meters. Used in calculating matrix
    :param save_metrics: (bool) save a dictionary with some spatial fire metrics to a JSON
    :param outname_metrics: (str) just name, not path
    :param outname: (str) name for output shapefile. just name, not path.
    :param outdir: (str) directory for all outputs.
    :return:
    """
    if not outdir:
        outdir = os.path.split(change_img)[0]
    with rasterio.open(change_img, 'r') as src:
        final_out = src.read()
        meta = src.meta
    final_out = np.where(final_out >= 1, 1, 0)
    meta['nodata'] = 0
    meta['dtype'] = 'int16'
    # from an online cookbook
    shapes = ({'properties': {'raster_val': v}, 'geometry': s}
              for i, (s, v)
              in enumerate(
        rasterio.features.shapes(final_out, mask=final_out.astype('bool'), transform=meta['transform'])))
    shapes_list = []
    for shape in shapes:
        shapes_list.append(shape)
    collection = geojson.FeatureCollection(shapes_list)

    # get the interiors
    change_gdf = gpd.GeoDataFrame.from_features(collection['features'], crs=meta['crs'])

    # simplifying change geometry
    changed_polys = []
    for poly in change_gdf.unary_union:
        changed_polys.append(poly)
    change_gdf = gpd.GeoDataFrame(changed_polys, columns=['geometry'], crs=meta['crs'])

    # get interiors
    interiors = change_gdf.interiors
    interior_rings = []
    for poly in interiors:
        if poly:
            interior_rings = interior_rings + poly

    # convert from LinearRings into Polygons
    interior_polys = []
    for ring in interior_rings:
        interior_polys.append(Polygon(ring))

    # convert from list to GeoDataFrame
    interiors_gdf = gpd.GeoDataFrame(interior_polys, columns=['geometry'], crs=meta['crs'])

    change_buff = change_gdf.buffer(buffer_size)
    change_debuff = change_buff.buffer(-1*buffer_size)
    change_debuff = change_debuff.unary_union

    # convert multipolygon into geoseries
    tmp_df = gpd.GeoDataFrame(columns=['geometry'])
    for geom in range(len(change_debuff)):
        tmp_df.loc[geom, 'geometry'] = change_debuff[geom]

    # turn into a GeoDataFrame
    matrix_extended = gpd.GeoDataFrame(tmp_df, columns=['geometry'], crs=meta['crs'])

    # difference with interiors
    matrix_mp = matrix_extended['geometry'].difference(interiors_gdf.unary_union)

    # difference with changed areas
    matrix_mp = matrix_mp.buffer(0).difference(change_gdf.unary_union.buffer(0))

    matrix_gdf = gpd.GeoDataFrame(matrix_mp, columns=['geometry'], crs=meta['crs'])
    matrix_gdf = matrix_gdf.dropna(how='any', axis=0)  # end up with some None values otherwise

    # simplify the geometry
    tmp = []
    for poly in matrix_gdf.unary_union:
        if poly:
            tmp.append(poly)
    matrix_gdf = gpd.GeoDataFrame(tmp, columns=['geometry'], crs=meta['crs'])

    change_gdf['classification'] = 'disturbed'
    interiors_gdf['classification'] = 'undisturbed'
    matrix_gdf['classification'] = 'matrix'
    combined_gdf = gpd.GeoDataFrame(pd.concat([change_gdf, interiors_gdf, matrix_gdf]), crs=meta['crs'])
    combined_gdf.to_file(os.path.join(outdir, outname))

    if save_metrics:
        # prepare to export all these metrics for later
        largest_patch = change_gdf.area.max() / 10000.
        num_patch = len(change_gdf)
        num_island = len(interiors_gdf)
        rdpd = len(change_gdf) / (change_gdf.area.sum() / 1000000.)  # num disturbed patches per 100 ha of event
        ldpi = change_gdf.area.max() / change_gdf.area.sum()
        event_area = (change_gdf.area.sum() + interiors_gdf.area.sum() + matrix_gdf.area.sum()) / 10000.
        frac_change = change_gdf.area.sum() / event_area / 10000.
        frac_islands = interiors_gdf.area.sum() / event_area / 10000.
        frac_matrix = matrix_gdf.area.sum() / event_area / 10000.
        # shape index = (perimeter/area) / (perimeter of square with area of EA)
        shape_index = combined_gdf.unary_union.length / (4*np.sqrt(combined_gdf.unary_union.area))

        # stick them all in a dictionary to export to json
        metrics = dict()
        metrics["largest_patch"] = largest_patch
        metrics["num_patch"] = num_patch
        metrics["num_island"] = num_island
        metrics["rdpd"] = rdpd
        metrics["ldpi"] = ldpi
        metrics["event_area"] = event_area
        metrics["frac_change"] = frac_change
        metrics["frac_islands"] = frac_islands
        metrics["frac_matrix"] = frac_matrix
        metrics["shape_index"] = shape_index
        # save to a JSON
        with open(os.path.join(outdir, outname_metrics), 'w') as fp:
            json.dump(metrics, fp)

    return os.path.join(outdir, outname)


def check_segs_for_decrease(sorted_dates, segs_arr, keepers_arr, start_date="20170528", end_date="20171005",
                            ref_date_str="20160101"):
    """

    :param sorted_dates:
    :param segs_arr:
    :param keepers_arr:
    :param start_date:
    :param end_date:
    :param ref_date_str: date from which all dates in sorted_dates are counted. You probably don't want to change this
    :return:
    """
    ref_date_obj = datetime.date(year=int(ref_date_str[0:4]), month=int(ref_date_str[4:6]), day=int(ref_date_str[6:8]))
    start_date_obj = datetime.date(year=int(start_date[0:4]), month=int(start_date[4:6]), day=int(start_date[6:8]))
    end_date_obj = datetime.date(year=int(end_date[0:4]), month=int(end_date[4:6]), day=int(end_date[6:8]))

    date_objects_sorted = []
    start_idx = []
    end_idx = []
    for i, day in enumerate(sorted_dates):
        date_objects_sorted.append(ref_date_obj + datetime.timedelta(days=int(day)))
        if date_objects_sorted[i] == start_date_obj:
            start_idx = i  # index in sorted_dates corresponding to the starting date in our analysis
        if date_objects_sorted[i] == end_date_obj:
            end_idx = i  # index in sorted_dates corresponding to the ending date in our analysis
    # if these did not get defined, then we need to break
    assert start_idx
    assert end_idx
    start_date_val = sorted_dates[start_idx]  # integer, counted up from first day of sorted_dates
    end_date_val = sorted_dates[end_idx]

    dates = sorted_dates[keepers_arr.astype('bool')]
    if len(dates) <= 3:  # error handling for pixels where the series is filled with 0s or too short to fit segs
        return 0
    if dates[0] > start_date_val:  # if our first usable image is after start_date
        return 0  # return 0
    x = np.arange(dates[0], dates[-1])  # every day from the beginning to the end of 'dates'
    y = np.interp(x, dates, segs_arr)  # linear interpolation at daily frequency. Essentially makes segments 'concrete'

    idx_start = int(np.argwhere(x == start_date_val))  # index of the segments corresponding to our start date
    idx_end = int(np.argwhere(x == end_date_val))
    delta = y[idx_end] - y[idx_start]  # ndvi of the point on the segment
    return delta
