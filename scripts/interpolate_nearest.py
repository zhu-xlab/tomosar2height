import numpy as np
import pandas as pd
import laspy
import scipy.spatial
import rasterio
from rasterio.transform import from_origin

# Step 1: Load the LAS file and extract the points
las_file = "/workspace/projects/tomosar2height/data/berlin/cloud/chunk_011-input_point_cloud.las"
output_tif = "berlin_zmax_nearest.tif"
resolution = 1.0
epsg = 25833

las = laspy.read(las_file)

# Extract X, Y, Z coordinates
x, y, z = las.x, las.y, las.z

# Step 2: Create a DataFrame and find the maximum Z for each X, Y coordinate
points = np.vstack((x, y, z)).T
df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
print("Grouping by X, Y and finding maximum Z value...")
max_z_df = df.groupby(['X', 'Y'], as_index=False).max()

# Step 3: Create a grid for interpolation
resolution = 1.0  # Adjust the grid resolution as needed
grid_y, grid_x = np.mgrid[
    max_z_df['Y'].min():max_z_df['Y'].max():resolution,
    max_z_df['X'].min():max_z_df['X'].max():resolution
]

# Step 4: Use KDTree for faster nearest neighbor interpolation
print("Performing nearest neighbor interpolation...")
tree = scipy.spatial.cKDTree(max_z_df[['X', 'Y']].values)
distances, indices = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], workers=-1)
grid_z = max_z_df['Z'].values[indices].reshape(grid_x.shape)

# Step 5: Correct the origin and save as GeoTIFF

# Use the minimum X and minimum Y as the origin for the top-left corner
origin_x = max_z_df['X'].min()
origin_y = max_z_df['Y'].min()
transform_matrix = from_origin(origin_x, origin_y, resolution, -resolution)

print("Saving the height map as a GeoTIFF...")
with rasterio.open(
    output_tif,
    'w',
    driver='GTiff',
    height=grid_z.shape[0],
    width=grid_z.shape[1],
    count=1,
    dtype=grid_z.dtype,
    crs=f"EPSG:{epsg}",  # Confirm CRS: EPSG:25833 for UTM zone 33N, ETRS89
    transform=transform_matrix,
) as dst:
    dst.write(grid_z, 1)

print(f"Height map saved to {output_tif}")
