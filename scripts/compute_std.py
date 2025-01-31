import laspy
import numpy as np

# Load the LAS file
las_file_path = '../data/munich/cloud/ps_utm_munich_coh60_urban_filter_noheader.las'
las_file = laspy.read(las_file_path)

# Extract Z coordinates
z_coordinates = las_file.z

# Calculate percentiles
percentile_5th = np.percentile(z_coordinates, 5)
percentile_95th = np.percentile(z_coordinates, 95)

# Filter Z coordinates within the specified percentile range
filtered_z_coordinates = z_coordinates[(z_coordinates >= percentile_5th) & (z_coordinates <= percentile_95th)]

# Compute standard deviation
std_deviation = np.std(filtered_z_coordinates)

print(f"Standard Deviation of Z coordinates from 5th to 95th percentile: {std_deviation}")
