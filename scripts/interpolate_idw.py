import numpy as np
import pandas as pd
import laspy
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree


def inverse_distance_weighting(x, y, z, grid_x, grid_y, power=2):
    tree = cKDTree(np.c_[x, y])
    dist, idx = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], k=8)

    # Directly assign values where the distance is zero
    weights = np.zeros_like(dist)
    zero_dist_mask = (dist == 0)
    weights[zero_dist_mask] = 1  # If distance is zero, assign a weight of 1 to that point

    # Calculate inverse distance weights for non-zero distances
    non_zero_dist_mask = ~zero_dist_mask
    weights[non_zero_dist_mask] = 1 / (dist[non_zero_dist_mask] ** power)

    # Normalize the weights
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights /= weights_sum

    interpolated = np.sum(weights * z[idx], axis=1)
    return interpolated.reshape(grid_x.shape)

# Step 1: Load the LAS file and extract the points
las_file = "/workspace/projects/tomosar2height/data/berlin/cloud/chunk_011-input_point_cloud.las"
output_tif = "berlin_zmax_idw.tif"
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
grid_x, grid_y = np.meshgrid(
    np.arange(max_z_df['X'].min(), max_z_df['X'].max(), resolution),
    np.arange(max_z_df['Y'].min(), max_z_df['Y'].max(), resolution)
)

# Step 4: Perform IDW interpolation
print("Performing IDW interpolation...")
grid_z = inverse_distance_weighting(
    max_z_df['X'].values, max_z_df['Y'].values, max_z_df['Z'].values, grid_x, grid_y
)

# Step 5: Correct the origin and save as GeoTIFF
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
    crs=f"EPSG:{epsg}",
    transform=transform_matrix,
) as dst:
    dst.write(grid_z, 1)

print(f"Height map saved to {output_tif}")
