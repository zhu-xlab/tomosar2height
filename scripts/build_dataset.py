import logging
import os
import shutil
import sys
from typing import List, Dict
from collections import defaultdict

import numpy as np
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from rasterio.windows import Window

from utils import lock_seed
from utils import load_pc, save_pc_to_ply
from utils import crop_pc_2d
from utils import RasterReader
from utils import dilate_mask


@hydra.main(config_path='../conf', config_name='config', version_base='1.2')
def build(cfg: DictConfig):

    # Shorthands
    build_training_data = cfg.get('build_training_data', False)
    cfg_chunk = cfg['chunk']

    input_pc_merged = cfg.get('input_pointcloud_merged', None)
    input_pc_folder = cfg.get('input_pointcloud_folder', None)
    if input_pc_merged is not None:
        # If exist merged point cloud, use merged one
        input_pc_paths: List = [input_pc_merged]
    elif input_pc_folder is not None:
        input_pc_paths: List = [
            os.path.join(input_pc_folder, _path) for _path in os.listdir(input_pc_folder)
        ]
    else:
        logging.error("No input point cloud.")
        raise IOError("No input point cloud.")

    cfg_output = cfg['output']
    output_folder = cfg_output['output_folder']
    save_vis = cfg_output['save_visualization_pc']

    # lock seed
    if cfg['lock_seed']:
        lock_seed(0)

    # %% Generate chunks
    chunk_x = cfg_chunk['chunk_x']
    chunk_y = cfg_chunk['chunk_y']
    chunk_bound = np.array([min(chunk_x), min(chunk_y), max(chunk_x), max(chunk_y)])
    chunks: Dict[int, Dict] = defaultdict(Dict)
    for i, x_l in enumerate(chunk_x[:-1]):
        for j, y_b in enumerate(chunk_y[:-1]):
            _p_min = np.array([x_l, y_b])
            _p_max = np.array([chunk_x[i + 1], chunk_y[j + 1]])
            chunks[len(chunks)] = {'min_bound': _p_min, 'max_bound': _p_max}

    # Clear target directory
    if os.path.exists(output_folder):
        _remove_old = input(f"Output folder exists at '{output_folder}', \n\r remove old one? (y/n): ")
        if 'y' == _remove_old:
            try:
                shutil.rmtree(output_folder)  # force remove
                logging.info(f"Removed old output folder: '{output_folder}'")
            except OSError as e:
                logging.error(e)
                logging.error("Build failed. Remove output folder manually and try again")
                sys.exit()
        if 'n' == _remove_old:
            logging.info("Remove output folder manually and try again")
            sys.exit()

    # Create folders
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    logging.info(f"Output folder ready at: '{output_folder}'")

    # Load point clouds and merge
    merged_pts: np.ndarray = np.empty((0, 3))
    for _full_path in tqdm(input_pc_paths, desc="Loading point clouds"):
        _temp_points = load_pc(_full_path)
        merged_pts = np.append(merged_pts, _temp_points, 0)

    del _temp_points
    logging.info("Point clouds merged")

    # load masks
    mask_keys = ['building']
    cfg_mask_files = cfg['mask_files']
    raster_masks: Dict[str, RasterReader] = {key: RasterReader(cfg_mask_files[key]) for key in mask_keys
                                             if cfg_mask_files[key] is not None}
    dsm_gt: RasterReader = RasterReader(cfg['gt_dsm'])

    # dilate building mask
    dilate_build = cfg.get('dilate_building', None)
    if dilate_build is not None:
        _mask = raster_masks['building'].get_data()
        _mask = dilate_mask(_mask, iterations=dilate_build)
        raster_masks['building'].set_data(_mask)

    out_of_mask_value = cfg['out_of_mask_value']
    logging.info("Raster masks loaded")

    # %% Main part
    # initialize
    chunk_safe_padding = cfg_chunk['chunk_safe_padding']
    chunk_info = defaultdict(dict)

    # Split data for each chunk
    for _chunk_idx in tqdm(chunks.keys(), desc="Chunks"):
        chunk_name = f"chunk_{_chunk_idx:03d}"
        chunk_dir = os.path.join(output_folder, chunk_name)
        os.makedirs(chunk_dir)
        _chunk_p1, _chunk_p2 = chunks[_chunk_idx]['min_bound'], chunks[_chunk_idx]['max_bound']
        chunk_info[_chunk_idx].update({
            'name': chunk_name,
        })
        if save_vis:
            vis_dir = os.path.join(chunk_dir, "vis")
            os.makedirs(vis_dir)

        # load DSM GT raster
        dsm_dim = dsm_gt.dataset_reader.height, dsm_gt.dataset_reader.width
        msk_dim = raster_masks['building'].dataset_reader.height, raster_masks['building'].dataset_reader.width
        if build_training_data:
            # get chunk bounding box
            _chunk_p1_pad = _chunk_p1 - np.array([chunk_safe_padding, chunk_safe_padding])
            _chunk_p2_pad = _chunk_p2 + np.array([chunk_safe_padding, chunk_safe_padding])
            _chunk_p1_pad = np.maximum(_chunk_p1_pad, chunk_bound[:2])
            _chunk_p2_pad = np.minimum(_chunk_p2_pad, chunk_bound[2:])

            max_x_dsm, min_y_dsm = dsm_gt.dataset_reader.index(*_chunk_p1_pad)
            min_x_dsm, max_y_dsm = dsm_gt.dataset_reader.index(*_chunk_p2_pad)

            max_x_msk, min_y_msk = raster_masks['building'].dataset_reader.index(*_chunk_p1_pad)
            min_x_msk, max_y_msk = raster_masks['building'].dataset_reader.index(*_chunk_p2_pad)

            # assume raster cover all chunks
            assert min_x_dsm >= 0 and min_y_dsm >= 0 and min_x_msk >= 0 and min_y_msk >= 0
            assert max_x_dsm <= dsm_dim[0] and max_y_dsm <= dsm_dim[1] and max_x_msk <= msk_dim[0] and max_y_msk <= msk_dim[1]

            window_dsm = Window.from_slices((min_x_dsm, max_x_dsm), (min_y_dsm, max_y_dsm))
            window_msk = Window.from_slices((min_x_msk, max_x_msk), (min_y_msk, max_y_msk))

            # read the data for the current window
            dsm_chunk = dsm_gt.dataset_reader.read(window=window_dsm)[0]
            raster_masks_chunk = {k: v.dataset_reader.read(window=window_msk)[0] for k, v in raster_masks.items()}
            dsm_chunk_min = dsm_chunk.min()
            dsm_chunk_max = dsm_chunk.max()
            # determine 3D bounding box
            if dsm_chunk_min < -1000 or dsm_chunk_max > 1000:
                # no-data value encountered
                logging.warning(f'invalid elevation value {dsm_chunk_min} ignored')
                dsm_chunk_min = dsm_chunk[dsm_chunk > -1000].min()
                dsm_chunk_max = dsm_chunk[dsm_chunk < 1000].max()
            chunk_p1_3d = np.array([*_chunk_p1, dsm_chunk_min])
            chunk_p2_3d = np.array([*_chunk_p2, dsm_chunk_max])

            assert (abs(chunk_p1_3d[:2] - _chunk_p1) < 1e-5).all()
            assert (abs(chunk_p2_3d[:2] - _chunk_p2) < 1e-5).all()
            chunk_info[_chunk_idx].update({
                'min_bound': chunk_p1_3d.tolist(),
                'max_bound': chunk_p2_3d.tolist(),
            })

        else:
            chunk_info[_chunk_idx].update({
                'min_bound': _chunk_p1.tolist(),
                'max_bound': _chunk_p2.tolist(),
            })

        # Save input point cloud
        chunk_input_pc, _ = crop_pc_2d(merged_pts, _chunk_p1, _chunk_p2)
        _output_path = os.path.join(chunk_dir, 'input_point_cloud.npz')
        _out_data = {
            'pts': chunk_input_pc
        }
        np.savez(_output_path, **_out_data)

        if save_vis:
            _output_path = os.path.join(vis_dir, f"{chunk_name}-input_point_cloud.ply")
            save_pc_to_ply(_output_path, chunk_input_pc)

    # %% chunk_info.yaml
    _output_path = os.path.join(output_folder, "chunk_info.yaml")
    with open(_output_path, 'w+') as f:
        yaml.dump(dict(chunk_info), f, default_flow_style=None, allow_unicode=True, Dumper=Dumper)
    logging.info(f"chunk_info saved to: '{_output_path}'")


if __name__ == '__main__':
    build()
