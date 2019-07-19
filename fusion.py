"""
Fuse images from different sensors using the principals of multivariate alteration detection.

@author: Nick Leach
"""

import os
import rasterio
import rasterio.mask
import rasterio.features
import fiona
import numpy as np
from geopandas import GeoDataFrame
import pandas as pd
from shapely.geometry import shape
import datetime


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
    Creates a dictionary with some basic info about the input HLS image
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
    :param directory: (string)
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


# TODO: change this so that it takes in the dictionary of HLS images rather than just the datetime objects.
    # that will make it a lot easier to grab the correct image for the normalization process
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
    nearest_image = date_differences[idx_min][1]
    return nearest_image


