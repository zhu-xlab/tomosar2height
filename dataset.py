import logging
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import transformations
import yaml
from yaml import CLoader as Loader
from torch.utils import data
from tqdm import tqdm

from utils import crop_pc_2d
from utils import invert_transform, apply_transform
from utils import RasterReader, RasterData

# Define land types and their indices
LAND_TYPES = ['building']
LAND_TYPE_IDX = {LAND_TYPES[i]: i for i in range(len(LAND_TYPES))}

# Constants for data augmentation
_origin = np.array([0., 0., 0.])
_x_axis = np.array([1., 0., 0.])
_y_axis = np.array([0., 1., 0.])
z_axis = np.array([0., 0., 1.])

# Predefined rotation matrices (90-degree increments clockwise)
rot_mat_dic: Dict[int, torch.Tensor] = {
    0: torch.eye(4).double(),
    1: torch.as_tensor(transformations.rotation_matrix(-90. * math.pi / 180., z_axis)).double(),
    2: torch.as_tensor(transformations.rotation_matrix(-180. * math.pi / 180., z_axis)).double(),
    3: torch.as_tensor(transformations.rotation_matrix(-270. * math.pi / 180., z_axis)).double(),
}

# Predefined flip matrices
flip_mat_dic: Dict[int, torch.Tensor] = {
    -1: torch.eye(4).double(),
    0: torch.as_tensor(transformations.reflection_matrix(_origin, _x_axis)).double(),  # flip on x direction (x := -x)
    1: torch.as_tensor(transformations.reflection_matrix(_origin, _y_axis)).double()  # flip on y direction (y := -y)
}


class TomoSARDataset(data.Dataset):
    """
    PyTorch Dataset for processing TomoSAR data with optional
    satellite images, DSM data, and point cloud inputs.
    """

    # Predefined filenames
    INPUT_POINT_CLOUD = "input_point_cloud.npz"
    QUERY_POINTS = "query--%s.npz"
    CHUNK_INFO = "chunk_info.yaml"

    def __init__(self, split: str, cfg_dataset: Dict, random_sample=False,
                 random_length=None, flip_augm=False, rotate_augm=False):
        """
        Args:
            split (str): Dataset split ('train', 'val', 'test', 'vis').
            cfg_dataset (Dict): Configuration dictionary for the dataset.
            random_sample (bool): Whether to randomly sample patches.
            random_length (int): Number of random samples if random_sample=True.
            flip_augm (bool): Enable data augmentation via flipping.
            rotate_augm (bool): Enable data augmentation via rotation.
        """

        # Shortcuts
        self.split = split
        self._dataset_folder = cfg_dataset['path']
        self._cfg_data = cfg_dataset
        self.patch_size = torch.tensor(cfg_dataset['patch_size'], dtype=torch.float64)

        # Initialize data structures
        self.images: List[RasterData] = []
        self.data_dic = defaultdict()
        self.dataset_chunk_idx_ls: List = cfg_dataset[f"{split}_chunks"]

        # Load chunk information
        dataset_dir = self._cfg_data['path']
        with open(os.path.join(dataset_dir, self.CHUNK_INFO), 'r') as f:
            self.chunk_info: Dict = yaml.load(f, Loader=Loader)
        self.chunk_info_ls: List = [self.chunk_info[i] for i in self.dataset_chunk_idx_ls]

        # Load satellite images
        images_dic = self._cfg_data.get('satellite_image', None)
        if images_dic is not None:
            image_folder = images_dic['folder']
            for image_name in images_dic['pairs']:
                _path = os.path.join(image_folder, image_name)
                reader = RasterReader(_path)
                self.images.append(reader)
                logging.debug(f"Satellite image loaded: {image_name}")

            # Check constraints on the number of images
            assert len(self.images) <= 2, "Only support single image or stereo image"
            assert self.images[-1].T == self.images[0].T

            # Normalize image data
            temp_ls = []
            for _img in self.images:
                _img_arr_r = _img.get_data(1).astype(np.int32)[None, ::]
                _img_arr_g = _img.get_data(2).astype(np.int32)[None, ::]
                _img_arr_b = _img.get_data(3).astype(np.int32)[None, ::]
                temp_ls.append(torch.from_numpy(_img_arr_r))
                temp_ls.append(torch.from_numpy(_img_arr_g))
                temp_ls.append(torch.from_numpy(_img_arr_b))
            self.norm_image_data: torch.Tensor = torch.cat(temp_ls, 0).long()  # n_img x h_image x w_image
            self._image_mean = images_dic['normalize']['mean']
            self._image_std = images_dic['normalize']['std']
            self.norm_image_data: torch.Tensor = (self.norm_image_data.double() - torch.tensor(self._image_mean)[:,
                                                                                  None, None]) / torch.tensor(
                self._image_std)[:, None, None]

        # Determine image-related properties
        self.n_images = len(self.images)
        if self.n_images > 0:
            self._image_pixel_size = torch.as_tensor(self.images[0].pixel_size, dtype=torch.float64)
            self._image_patch_shape = self.patch_size / self._image_pixel_size
            assert torch.all(torch.floor(self._image_patch_shape) == self._image_patch_shape), \
                "Patch size should be integer multiple of image pixel size"
            self._image_patch_shape = torch.floor(self._image_patch_shape).long()

        # Load DSM data
        dsm_path = self._cfg_data.get('dsm_gt_path', None)
        dsm_reader = RasterReader(dsm_path)
        dsm_array = dsm_reader.get_data(1).astype(np.float32)
        self.dsm = dsm_reader
        self._dsm_pixel_size = torch.as_tensor(self.dsm.pixel_size, dtype=torch.float64)
        self.dsm_data = torch.tensor(dsm_array, dtype=torch.float32)
        self._dsm_patch_shape = self.patch_size / self._dsm_pixel_size
        assert torch.all(torch.floor(self._dsm_patch_shape) == self._dsm_patch_shape), \
            "Patch size should be integer multiple of DSM pixel size"
        self._dsm_patch_shape = torch.floor(self._dsm_patch_shape).long()

        # Load point cloud data by chunks
        for chunk_idx in tqdm(self.dataset_chunk_idx_ls, desc=f"Loading {self.split} data to RAM"):
            info = self.chunk_info[chunk_idx]
            chunk_name = info['name']
            chunk_full_path = os.path.join(dataset_dir, chunk_name)
            inputs = np.load(os.path.join(chunk_full_path, self.INPUT_POINT_CLOUD))
            chunk_data = {
                'name': chunk_name,
                'inputs': torch.from_numpy(inputs['pts']).double(),
            }
            self.data_dic[chunk_idx] = chunk_data

        # Random sampling parameters
        self.random_sample = random_sample
        self.random_length = random_length
        if self.random_sample and random_length is None:
            logging.warning("random_length not provided when random_sample = True")
            self.random_length = 10

        # Data augmentation parameters
        self.flip_augm = flip_augm
        self.rotate_augm = rotate_augm

        # Generate anchors for sliding window
        self.anchor_points: List[Dict] = []
        if not self.random_sample:
            self.slide_window_strip = cfg_dataset['sliding_window'][f'{self.split}_strip']
            for chunk_idx in self.dataset_chunk_idx_ls:
                chunk_info = self.chunk_info[chunk_idx]
                _min_bound_np = np.array(chunk_info['min_bound'])
                _max_bound_np = np.array(chunk_info['max_bound'])
                _chunk_size_np = _max_bound_np - _min_bound_np
                patch_x_np = np.arange(_min_bound_np[0], _max_bound_np[0] - self.patch_size[0],
                                       self.slide_window_strip[0])
                patch_x_np = np.concatenate([patch_x_np, np.array([_max_bound_np[0] - self.patch_size[0]])])
                patch_y_np = np.arange(_min_bound_np[1], _max_bound_np[1] - self.patch_size[1],
                                       self.slide_window_strip[1])
                patch_y_np = np.concatenate([patch_y_np, np.array([_max_bound_np[1] - self.patch_size[1]])])
                xv, yv = np.meshgrid(patch_x_np, patch_y_np)
                anchors = np.concatenate([xv.reshape((-1, 1)), yv.reshape((-1, 1))], 1)
                anchors = torch.from_numpy(anchors).double()
                for anchor in anchors:
                    self.anchor_points.append({
                        'chunk_idx': chunk_idx,
                        'anchor': anchor
                    })

        # Normalization factors
        _x_range = cfg_dataset['normalize']['x_range']
        _y_range = cfg_dataset['normalize']['y_range']
        self._min_norm_bound = [_x_range[0], _y_range[0]]
        self._max_norm_bound = [_x_range[1], _y_range[1]]
        self.z_bound = cfg_dataset['normalize']['z_bound']
        self.scale_mat = torch.diag(torch.tensor([self.patch_size[0] / (_x_range[1] - _x_range[0]),
                                                  self.patch_size[1] / (_y_range[1] - _y_range[0]),
                                                  self.z_bound[1] - self.z_bound[0],
                                                  1], dtype=torch.float64))
        # Shift from [-0.5, 0.5] to [0, 1]
        self.shift_norm = torch.cat([torch.eye(4, 3, dtype=torch.float64),
                                     torch.tensor([(_x_range[1] - _x_range[0]) / 2.,
                                                   (_y_range[1] - _y_range[0]) / 2., 0, 1]).reshape(-1, 1)], 1)

    def __len__(self):
        return self.random_length if self.random_sample else len(self.anchor_points)

    def __getitem__(self, idx):
        """
        Get a data patch, including normalized point cloud, optional satellite image, and DSM data.
        Args:
            idx (int): Index of the patch.

        Returns:
            dict: Contains normalized inputs, transformation matrix, and auxiliary data.
        """
        # Get patch anchor point
        if self.random_sample:
            chunk_idx = self.dataset_chunk_idx_ls[idx % len(self.dataset_chunk_idx_ls)]
            chunk_info = self.chunk_info[chunk_idx]
            _min_bound = torch.tensor(chunk_info['min_bound'], dtype=torch.float64)
            _max_bound = torch.tensor(chunk_info['max_bound'], dtype=torch.float64)
            _chunk_size = _max_bound - _min_bound
            _rand = torch.rand(2, dtype=torch.float64)
            anchor = _rand * (_chunk_size[:2] - self.patch_size[:2])
            if self.n_images > 0:
                anchor = torch.floor(anchor / self._image_pixel_size) * self._image_pixel_size
            anchor += _min_bound[:2]
        else:
            _anchor_info = self.anchor_points[idx]
            chunk_idx = _anchor_info['chunk_idx']
            anchor = _anchor_info['anchor']

        min_bound = anchor
        max_bound = anchor + self.patch_size.double()
        assert chunk_idx in self.dataset_chunk_idx_ls
        assert torch.float64 == min_bound.dtype  # for geo-coordinate, must use float64

        # Crop inputs
        chunk_data = self.data_dic[chunk_idx]
        inputs, _ = crop_pc_2d(chunk_data['inputs'], min_bound, max_bound)
        if len(inputs) == 0:
            return {
                'name': f"{chunk_data['name']}-patch{idx}",
                'min_bound': min_bound.double().clone(),
                'max_bound': max_bound.double().clone(),
                'is_valid': False  # will not be processed
            }

        # Determine z-axis shift strategy
        shift_strategy = self._cfg_data['normalize']['z_shift']
        if 'local_min' == shift_strategy:
            z_shift = torch.min(inputs[:, 2]).double().reshape(1)
        elif 'global_min' == shift_strategy:
            z_shift = torch.tensor([self.z_bound[0]])
        else:
            raise ValueError(f"Unknown shift strategy: {shift_strategy}")

        # Apply data augmentation
        if self.rotate_augm:
            rot_times = list(rot_mat_dic.keys())[np.random.choice(len(rot_mat_dic))]
        else:
            rot_times = 0
        rot_mat = rot_mat_dic[rot_times]

        if self.flip_augm:
            flip_dim_pc = list(flip_mat_dic.keys())[np.random.choice(len(flip_mat_dic))]
        else:
            flip_dim_pc = -1
        flip_mat = flip_mat_dic[flip_dim_pc]

        # Transformation matrix: normalize to [-0.5, 0.5]
        transform_mat = self.scale_mat.clone()
        transform_mat[0:3, 3] = torch.cat([(min_bound + max_bound) / 2., z_shift], 0)
        normalize_mat = self.shift_norm.double() @ flip_mat.double() @ rot_mat.double() \
                        @ invert_transform(transform_mat).double()
        transform_mat = invert_transform(normalize_mat)
        assert torch.float64 == transform_mat.dtype

        # Normalize inputs
        inputs_norm = apply_transform(inputs, normalize_mat)
        inputs_norm = inputs_norm.float()

        # Re-crop to ensure normalization consistency
        inputs_norm, _ = crop_pc_2d(inputs_norm, self._min_norm_bound, self._max_norm_bound)  # XYZ in [0, 1]

        out_data = {
            'name': f"{chunk_data['name']}-patch{idx}",
            'inputs': inputs_norm,
            'transform': transform_mat.double().clone(),
            'min_bound': min_bound.double().clone(),
            'max_bound': max_bound.double().clone(),
            'flip': flip_dim_pc,
            'rotate': rot_times,
            'is_valid': True
        }

        # Process satellite image
        if self.n_images > 0:
            # index of bottom-left pixel center
            _anchor_pixel_center = anchor + self._image_pixel_size / 2.
            _col, _row = self.images[0].query_col_row(_anchor_pixel_center[0], _anchor_pixel_center[1])
            shape = self._image_patch_shape
            image_tensor = self.norm_image_data[:, _row - shape[0] + 1:_row + 1,
                           _col:_col + shape[1]]  # n_img x h_patch x w_patch

            # Apply image augmentation
            if rot_times > 0:
                image_tensor = image_tensor.rot90(rot_times, [-1, -2])  # rotate clockwise
            if flip_dim_pc >= 0:
                if 0 == flip_dim_pc:  # points flip on x direction (along y), image flip columns
                    image_tensor = image_tensor.flip(-1)
                if 1 == flip_dim_pc:  # points flip on y direction (along x), image flip rows
                    image_tensor = image_tensor.flip(-2)

            assert torch.Size([3, shape[0], shape[1]]) == image_tensor.shape, f"chunk_idx:{chunk_idx}"
            out_data['image'] = image_tensor.float().flip(-2)

        # Process DSM data
        _anchor_pixel_center = anchor + self._dsm_pixel_size / 2.
        _col, _row = self.dsm.query_col_row(_anchor_pixel_center[0], _anchor_pixel_center[1])
        shape = self._dsm_patch_shape
        dsm_tensor = self.dsm_data[_row - shape[0] + 1:_row + 1, _col:_col + shape[1]]  # n_img x h_patch x w_patch

        # Apply DSM augmentation
        if rot_times > 0:
            dsm_tensor = dsm_tensor.rot90(rot_times, [-1, -2])  # rotate clockwise
        if flip_dim_pc >= 0:
            if 0 == flip_dim_pc:  # points flip on x direction (along y), image flip columns
                dsm_tensor = dsm_tensor.flip(-1)
            if 1 == flip_dim_pc:  # points flip on y direction (along x), image flip rows
                dsm_tensor = dsm_tensor.flip(-2)

        assert torch.Size([shape[0], shape[1]]) == dsm_tensor.shape, f"chunk_idx:{chunk_idx}"
        out_data['dsm'] = dsm_tensor.float().flip(-2)

        return out_data
