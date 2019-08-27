"""
Fuse images from different sensors using the principles of multivariate alteration detection.

@author: Nick Leach
"""

import os
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
    os.chdir(r"C:\Users\nleach\PycharmProjects\untitled")
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
        empty = np.zeros_like(src.read(1))
        for shp, val in rasterio.features.shapes(source=empty, transform=src.transform):
            d['val'].append(val)
            geometry.append(shape(shp))
        raster_crs = src.crs
    df = pd.DataFrame(data=d)
    geo_df = GeoDataFrame(df, crs={'init': raster_crs['init']}, geometry=geometry)
    geo_df['area'] = geo_df.area
    geo_df.to_file(outpath, driver="ESRI Shapefile")
    return


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
    Get basic info about every HLS image in directory.
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
        for f in [f for f in all_filepaths if f.endswith("merge.tif")]:
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
    date_differences = np.asarray(date_differences)  # TODO: UNTESTED conversion
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


def fit_time_series(hls_dir, ps_dir=None, despike_thresh=0.10, max_segs=10, seg_thresh=0.05, start_year=2016,
                    nodata_val=-1000, verbose=True):
    """
    Loads in all the HLS arrays in hls_dir. For each pixel location, the NDVI is calculated and corresponding dates
    extracted. These values are then despiked and fit with linear segments.
    :param hls_dir:
    :param ps_dir:
    :param despike_thresh:
    :param max_segs:
    :param seg_thresh:
    :param start_year:
    :param nodata_val:
    :param verbose:
    :return all_pixel_data: tuple with (ndvi, despiked, segs, keepers)
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
    for img in hls_fileinfo:
        series_dates.append(hls_fileinfo[img]['doy'] + (hls_fileinfo[img]['year']-start_year)*365)
    if ps_dir:
        ps_dates = []
        for img in ps_fileinfo:
            ps_dates.append(ps_fileinfo[img]['doy'] + (ps_fileinfo[img]['year']-start_year)*365)
        series_dates = series_dates + ps_dates
    # Need to keep track of which dates belong to HLS and which belong to PS
    sorted_dates = sorted(series_dates)  # want a sorted version since the dates may initially be grouped by sensor

    # load in all the HLS arrays
    hls_arrays = []
    for dirpath, dirnames, filenames in os.walk(hls_dir):
        for img in [f for f in filenames if f.endswith(".tif")]:
            img_path = os.path.join(dirpath, img)
            with rasterio.open(img_path) as src:
                hls_arrays.append(src.read())

    # get dimensions for the HLS images (should all have identical dimensions)
    bands = hls_arrays[0].shape[0]
    rows = hls_arrays[0].shape[1]
    cols = hls_arrays[0].shape[2]

    all_pixel_data = []
    for row in range(rows):
        row_data = []
        for col in range(cols):
            series = []
            for band in range(bands):
                # read in the corresponding series (one pixel through all time and bands)
                series.append([arr[band][row][col] for arr in hls_arrays])
            ndvi = np.array((np.array(series[3]) - np.array(series[2])) / (
                        np.array(series[3]) + np.array(series[2])))  # ndvi of array
            del series
            # sort the ndvi series based on dates (remember, dates are originally unorderd - L30 then S30)
            ndvi = np.array([x for y, x in sorted(zip(series_dates, ndvi))])
            # remove dates where the pixel has been masked/satured
            keepers = []
            for pix in ndvi:
                if pix == nodata_val:
                    keepers.append(False)
                else:
                    keepers.append(True)
            keepers = np.array(keepers)
            # add the ndvi series for this position to a list of arrays
            # also add despiked values and segs
            # if (ndvi > 0) <= 3, just write some 0s. Otherwise, enter a try/except block
            if len(keepers[keepers > 0]) > 3:  # need at least 4 points to run the segment fitter
                try:
                    despiked = despike(ndvi[keepers], despike_thresh)[1]
                    segs = sf.seg_fit(despiked, max_segs, seg_thresh, np.array(sorted_dates)[keepers])
                    row_data.append((ndvi, despiked, segs, keepers))
                except:
                    print("Issue at row " + str(row) + " and col " + str(col))
                    row_data.append((ndvi, np.zeros_like(ndvi), np.zeros_like(ndvi), keepers))
            else:  # if there aren't enough points to segment fit, append all zeros
                print("Not enough valid data at row " + str(row) + " and col " + str(col))
                row_data.append((ndvi, np.zeros_like(ndvi), np.zeros_like(ndvi), keepers))
        all_pixel_data.append(row_data)
        if verbose:
            if row % 10 == 0:
                print("Finished row " + str(row) + " after " + str(int(time.time()-start)) + " seconds. ")
    end = time.time()
    print("Total time elapsed: " + str(int(end-start)) + " seconds.")
    return all_pixel_data
