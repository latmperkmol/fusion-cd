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
import osr
from osgeo.gdalconst import GDT_UInt16
import time
# weird and janky, but works until these libraries are combined
try:
    from despike import despike
    import segment_fitter as sf
except ModuleNotFoundError:
    sys.path.insert(1, r"C:\Users\nleach\PycharmProjects\untitled")
    from despike import despike
    import segment_fitter as sf


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


def read_hdf_to_arr(hdf_path, band, datatype=np.int16):
    """
    Functionalizing the process of reading HDF files into arrays
    read a single band out of the hdf and load it into a numpy array
    """
    if os.path.isfile(hdf_path):
        src = gdal.Open(hdf_path)
        band_ds = gdal.Open(src.GetSubDatasets()[band][0], gdal.GA_ReadOnly)
        band_array = band_ds.ReadAsArray().astype(datatype)
        del src
        return band_array
    else:
        print("That file does not exist")
        return


def get_hdf_transform(hdf_path):
    if os.path.isfile(hdf_path):
        src = gdal.Open(hdf_path)
        band_ds = gdal.Open(src.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
        prj = band_ds.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        return prj, srs
    else:
        print("That file does not exist")
        return


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
    return


def make_shapefile_from_raster(raster, outname="vectorized.shp", outdir=None):
    """
    Generate a shapefile with a single feature outlining the extent of the input raster.
    There is probably a better way to do this, but this works...

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


def get_ncp(mad_img, save_image=False, out_path=None):
    """
    Calculate the no-change probability of each pixel in an input MAD image.
    :param mad_img: (string) path to the MAD image
    :param save_image: (bool) whether or not to save the NCP values as a single band GeoTiff
    :param out_path: (string) output destination of the NCP image
    :return:
    """
    from scipy.stats import chi2
    # read in the MAD image's chisquared band (last band) and ravel it
    with rasterio.open(mad_img, 'r') as src:
        # get info about the image (rows, cols, bands)
        profile = src.meta
        bands = profile["count"]
        chisqr = src.read(bands)

    # generate no-change probability for each pixel (location on normalized chi2 cdf - basically alpha value)
    chisqr_flat = chisqr.ravel()
    ncp = 1 - chi2.cdf(chisqr_flat, bands-1)
    # ncp = no-change probability, so low number means low prob of being unchanged. i.e. high numbers are invariant
    ncp_reshape = np.reshape(ncp, np.shape(chisqr))
    # if requested, save a new image with the ncp for each image
    if save_image:
        # if path is not specified, give it a silly name and save it with the input MAD image
        if not out_path:
            out_dir = os.path.split(mad_img)[0]
            out_path = os.path.join(out_dir, "ncp_img.tif")
        profile["count"] = 1
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(ncp_reshape)
    return ncp_reshape


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
    tgt_resolution = tgt_meta['transform'][0]  # in meters. Assumes square pixels
    with rasterio.open(reference_image, 'r') as src:
        ref_meta = src.profile
    dst_res = ref_meta['transform'][0]   # destination resolution in meters. Assumes square pixels
    left = ref_meta['transform'][2]
    top = ref_meta['transform'][5]
    right = left + ref_meta['width']*dst_res
    bottom = top + ref_meta['height']*dst_res
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
            is_hls = np.array([x for y, x in sorted(zip(series_dates, is_hls))])  # sorting is_hls based on dates
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
                          include_ps=True, row_num=None, col_num=None):
    """
    Given a time series for a single (pixel) location, calculate the despiked values and segment values.
    Requires the dates associated with each image and a boolean array stating which values to keep (True) and to ignore.
    If the input values come from a mix of HLS and PS, this can also use only the HLS values.
    :param ndvi:
    :param keepers:
    :param sorted_dates:
    :param despike_thresh:
    :param max_segs:
    :param seg_thresh:
    :param is_hls:
    :param include_ps:
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
        except:  # TODO: fix this naked except since it makes it difficult to kill the kernel when running
            print("Issue at row " + str(row_num) + " and column " + str(col_num))
            despiked = np.zeros_like(ndvi)
            segs = np.zeros_like(ndvi)
    else:
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
        arvi = ma.masked_outside(((nir - 2 * red + blue) / (nir + blue)) * msk / 255., -1.0, 1.0)
        varig = ma.masked_outside(((green - red) / (green + red - blue)) * msk / 255., -1.0, 1.0)
        gli_b = ma.masked_outside((((blue - red) + (blue - green)) / ((2 * blue) + (red + green))) * msk / 255., -1.0,
                                  1.0)
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
    :param dates: need to use the dates corresponding to each point in segs. i.e. not the fully sorted_dates array
    :return:
    """
    if len(segs >= 3):
        slope = (segs[1:]-segs[:-1])/(dates[1:]-dates[:-1])
        derivative = slope[1:]-slope[:-1]  # approximate the slope
        # make an array where dates representing breakpoints are marked as True and all other points are marked as False
        # same length as segs array
        breakpoints = abs(derivative) > 1e-10  # if derivative of segs is significantly greater than 0, then mark change
        breakpoints = np.insert(breakpoints, 0, False)  # buffer with a False value at the beginning
        breakpoints = np.append(breakpoints, False)  # Buffer with a False value at the end too
        if dates[breakpoints] > 0:
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
