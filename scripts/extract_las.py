import glob
import os
import arcpy
from tqdm import tqdm
from pathlib import Path
import multiprocessing
import laspy
import trimesh

# Define paths
las_path = 'C://Users/zhaiyu/Downloads/ps_utm_munich_coh60_urban_filter_noheader_translated.las'
obj_dir = 'C://Users/zhaiyu/Downloads/instances_globalCRS/instances_globalCRS/'
output_dir = 'C://Users/zhaiyu/Downloads/instances_output/'
temp_dir = 'C://Users/zhaiyu/Downloads/instances_output/temp/'

# Set environment settings
arcpy.env.workspace = "C://Users/zhaiyu/Downloads/instances_output"
num_workers = 11

# Calculate las extent
las_file = laspy.read(las_path)
las_head = las_file.header
las_bounds = [[*las_head.min], [*las_head.max]]


def process_one(obj_path):
    # Extract path variable
    obj_stem = Path(obj_path).stem

    # Skip the generated
    if os.path.exists(output_dir + f'ps_utm_munich_coh60_urban_filter_noheader_translated_{obj_stem}.las'):
        return

    # Skip non-overlapping instances
    mesh = trimesh.load(obj_path)
    mesh_bounds = mesh.bounds
    if (mesh_bounds[0][:2] < las_bounds[0][:2]).any() or (mesh_bounds[1][:2] > las_bounds[1][:2]).any():
        return

    # Import OBJ file as multipatch
    if not os.path.exists(temp_dir + f'{obj_stem}_3d.shp') and not os.path.exists(
            output_dir + f'ps_utm_munich_coh60_urban_filter_noheader_translated_{obj_stem}.las'):
        arcpy.ddd.Import3DFiles(in_files=obj_path, out_featureClass=temp_dir + f'{obj_stem}_3d.shp', file_suffix='OBJ')

    # Convert multipatch to footprint
    if not os.path.exists(temp_dir + f'{obj_stem}_2d.shp') and not os.path.exists(
            output_dir + f'ps_utm_munich_coh60_urban_filter_noheader_translated_{obj_stem}.las'):
        arcpy.ddd.MultiPatchFootprint(in_feature_class=temp_dir + f'{obj_stem}_3d.shp',
                                      out_feature_class=temp_dir + f'{obj_stem}_2d.shp')

    # Create buffer around footprint
    if not os.path.exists(temp_dir + f'{obj_stem}_2dbuffer.shp') and not os.path.exists(
            output_dir + f'ps_utm_munich_coh60_urban_filter_noheader_translated_{obj_stem}.las'):
        arcpy.Buffer_analysis(temp_dir + f'{obj_stem}_2d.shp', out_feature_class=temp_dir + f'{obj_stem}_2dbuffer.shp',
                              buffer_distance_or_field="2 Meters", line_side='FULL')

    # Extract point cloud within footprint
    if not os.path.exists(output_dir + f'ps_utm_munich_coh60_urban_filter_noheader_translated_{obj_stem}.las'):
        arcpy.ddd.ExtractLas(in_las_dataset=temp_dir + 'las_dataset.lasd', target_folder=output_dir,
                             boundary=temp_dir + f'{obj_stem}_2dbuffer.shp', name_suffix=f'_{obj_stem}')

    # remove temp files
    os.remove(temp_dir + f'{obj_stem}_3d.shp')
    os.remove(temp_dir + f'{obj_stem}_2d.shp')


if __name__ == '__main__':

    # Create LAS Dataset
    if not os.path.exists(temp_dir + 'las_dataset.lasd'):
        arcpy.management.CreateLasDataset(input=las_path, out_las_dataset=temp_dir + 'las_dataset.lasd')

    # List all OBJ files
    args = glob.glob(obj_dir + '*.obj')

    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_one, args), total=len(args)):
            pass
