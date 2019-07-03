import os
import numpy as np
import rasterio
from rasterio import plot
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# let's grab a random L8 image that I have lying around and play with that.
# image is a 4-band composite (B2,B3,B4,B5) clipped to a small area.

l8_path = r"E:\Leach\2017_fires_development_database\AFRF\publication_data\planet_images_cloud_free_cleaner\20170623_0f18_analytic_sr_merge\LC08_L1TP_047024_20160803_20170221_01_T1_trimmed.tif"

with rasterio.open(l8_path) as l8_ds:
    data_arr = l8_ds.read()
    kwargs = l8_ds.meta
    #rasterio.plot.show(l8_ds)

bands = kwargs['count']
data_vector = np.zeros((bands, np.size(data_arr[0])))
for b in range(len(data_arr)):
    data_vector[b] = np.ravel(data_arr[b])

pca = PCA()
PCs = pca.fit_transform(np.transpose(data_vector))
eigenvectors = pca.components_

# visualize the first PC mode
rows = kwargs['height']
cols = kwargs['width']
plt.imshow(np.reshape(PCs[:, 0], (rows, cols)))

# put each PC into its own band
pc_image = np.zeros((bands, rows, cols))
for b in range(bands):
    band_max = np.max(PCs[:, b]) # need to scale in order to visualize with rasterio.plot
    pc_image[b, :, :] = np.reshape(PCs[:, b], (rows, cols))/band_max

rasterio.plot.show(pc_image[0:3], title="PC-space image")

fig = plt.figure()
plt.title("pixels in PC-space")
plt.scatter(PCs[:, 0], PCs[:, 1], s=2, c=PCs[:, 2], marker='.', alpha=0.25)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
cbar = plt.colorbar()
cbar.set_label("PC 3")
plt.show()

fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PCs[:,0], PCs[:,1], PCs[:,2], s=3, marker='.')

# do a similar process, but for two Planet images (both trimmed to the same area and with the same no-data pixels)
# then, take the difference between the images in PC-space. Will create a 4-element PC-diff.
    # may need the higher order PCs in order to capture changes from burns
# can map the change vector for each pixel in 2,3, or 4D PC-space.
# Do this process for a) the full images, b) the images with burned areas masked out, and c) only the burned areas!!
# can also just look at the burned areas in PC space in the full images

# start by taking snips
import fiona
import rasterio.mask
afrf_bbox = r"E:\Leach\2017_fires_development_database\AFRF\field_data\bbox\burned_area_bbox.shp"
p1_path = r"E:\Leach\2017_fires_development_database\AFRF\publication_data\planet_images_cloud_free_cleaner\20170629_1030_analytic_sr_merge\20170629_1030_analytic_sr_merge_FINAL.tif"
with fiona.open(afrf_bbox, 'r') as shapefile:
    features = [feature['geometry'] for feature in shapefile]
with rasterio.open(p1_path, 'r') as src:
    out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
    out_meta = src.meta.copy()
out_meta.update({"height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

p1_snip_path = p1_path[:-4] + "_snip.tif"
with rasterio.open(p1_snip_path, 'w', **out_meta) as dst:
    dst.write(out_image)



