import yaml
import os
import time
import math
import rasterio
from rasterio.windows import Window


def split_chunks(input_path, output_dir, chunk_info):
    """
    Split chunks from a GeoTIFF image.
    """
    # Open the input GeoTIFF file
    with rasterio.open(input_path) as src:
        for chunk_id, chunk_data in chunk_info.items():
            # Extract bounds for the current chunk
            # an example:
            #   max_bound: [687165.0, 5335624.0, 577.19]
            #   min_bound: [686165.5, 5334624.0, 517.19]
            min_bound = chunk_data['min_bound']
            max_bound = chunk_data['max_bound']

            # Calculate the window for the current chunk
            max_x, min_y = src.index(*min_bound)
            min_x, max_y = src.index(*max_bound)

            window = Window.from_slices((min_x, max_x), (min_y, max_y))

            # Read the data for the current window
            output_data = src.read(window=window)

            # Define the output path for the current chunk
            output_path = os.path.join(output_dir, f'{chunk_data["name"]}.tif')

            # Write the chunk data to the output GeoTIFF file
            with rasterio.open(output_path, 'w', driver='GTiff', width=window.width, height=window.height, count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=src.window_transform(window)) as dst:
                dst.write(output_data)


def split_patches(chunk_path, output_dir, chunk_name, patch_size=512):
    """
    Split patches from a GeoTIFF chunk.
    """
    # Open the input GeoTIFF chunk file
    with rasterio.open(chunk_path) as src:
        # Get the dimensions of the chunk
        width = src.width
        height = src.height

        # Calculate the number of patches in each dimension
        num_patches_x = math.ceil(width / patch_size)
        num_patches_y = math.ceil(height / patch_size)

        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Calculate the window for the current patch
                min_y = i * patch_size
                max_y = min((i + 1) * patch_size, height)
                min_x = j * patch_size
                max_x = min((j + 1) * patch_size, width)

                window = Window.from_slices((min_y, max_y), (min_x, max_x))

                # Read the data for the current window
                output_data = src.read(window=window)

                # Define the output path for the current patch
                patch_name = f'{chunk_name}_patch_{i * num_patches_x + j}.tif'
                output_path = os.path.join(output_dir, patch_name)

                # Write the patch data to the output GeoTIFF file
                with rasterio.open(output_path, 'w', driver='GTiff', width=window.width, height=window.height, count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=src.window_transform(window)) as dst:
                    dst.write(output_data)


if __name__ == "__main__":
    SPLIT_CHUNKS = True
    SPLIT_PATCHES = True

    # Read chunk info from the configuration file
    with open('./data/TomoCity_filtered/generated/chunk_info.yaml', 'r') as file:
        chunk_info = yaml.safe_load(file)

    # Path to the input TIFF image
    input_path = './data/TomoCity_filtered/raster/dsm_lidar_50cm.tif'

    # Directory to save the output chunks
    output_dir = './outputs/munich_chunks_and_patches/gt_split'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the TIFF image into chunks
    if SPLIT_CHUNKS:
        print('Splitting chunks...')
        split_chunks(input_path, output_dir, chunk_info)
        time.sleep(3)

    if SPLIT_PATCHES:
        print('Splitting patches...')
        for chunk_id, chunk_data in chunk_info.items():
            chunk_path = os.path.join(output_dir, f'{chunk_data["name"]}.tif')
            chunk_output_dir = os.path.join(output_dir, f'{chunk_data["name"]}_patches')
            os.makedirs(chunk_output_dir, exist_ok=True)
            split_patches(chunk_path, chunk_output_dir, chunk_data["name"])
