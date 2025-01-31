import logging
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.io_raster import RasterWriter, RasterData
from dataset import TomoSARDataset


class DSMGenerator:
    NODATA_VALUE = np.nan

    def __init__(
            self, model: nn.Module, device, data_loader: DataLoader, dsm_pixel_size, fill_empty=False,
            h_range=None, h_res_0=0.25, upsample_steps=3, points_batch_size=300000, half_blend_percent=None,
            crs_epsg=25832, use_cloud=True, use_image=True, use_footprint=False
    ):
        self.model: nn.Module = model
        self.device = device
        self.fill_empty = fill_empty
        self.data_loader: DataLoader = data_loader
        self.pixel_size = torch.tensor(dsm_pixel_size, dtype=torch.float64)
        self.half_blend_percent = half_blend_percent or [0.5, 0.5]
        self.crs_epsg = crs_epsg
        self.h_range = torch.tensor(h_range or [-50, 100])
        self.h_res_0 = h_res_0
        self.upsample_steps = upsample_steps
        self.points_batch_size = points_batch_size
        self.use_cloud = use_cloud
        self.use_image = use_image
        self.use_footprint = use_footprint

        assert self.upsample_steps >= 1, "Upsample steps must be at least 1."

        self._dataset: TomoSARDataset = data_loader.dataset
        self.patch_size: torch.Tensor = self._dataset.patch_size.double()

        assert not self._dataset.random_sample, "Only regular patching is accepted."
        assert self.data_loader.batch_size == 1, "Only batch size == 1 is accepted."

        self._calculate_bounds()
        self.dsm_shape = RasterWriter.cal_dsm_shape(
            [self.l_bound, self.b_bound], [self.r_bound, self.t_bound], self.pixel_size
        )

        self._default_query_grid = self._generate_query_grid().to(self.device)
        self._default_true = torch.ones(self._default_query_grid.shape[:2], dtype=torch.bool).to(self.device)
        self._default_false = torch.zeros(self._default_query_grid.shape[:2], dtype=torch.bool).to(self.device)

        self.patch_weight = self._linear_blend_patch_weight(
            self._default_query_grid.shape[:2], self.half_blend_percent
        ).to(self.device)

        assert self.patch_weight.dtype == torch.float64, "Patch weight must be of type float64."

    def _calculate_bounds(self):
        self.l_bound = self.b_bound = np.inf
        self.r_bound = self.t_bound = -np.inf

        for info in self._dataset.chunk_info_ls:
            l, b = info['min_bound'][:2]
            r, t = info['max_bound'][:2]

            self.l_bound = min(self.l_bound, l)
            self.b_bound = min(self.b_bound, b)
            self.r_bound = max(self.r_bound, r)
            self.t_bound = max(self.t_bound, t)

    def _generate_query_grid(self):
        pzs = torch.arange(self.h_range[0].item(), self.h_range[1].item(), self.h_res_0)
        _grid_xy_shape = torch.round(self.patch_size / self.pixel_size).long()

        shape = [_grid_xy_shape[0].item(), _grid_xy_shape[1].item(), pzs.shape[0]]
        pxs = torch.linspace(0., 1., _grid_xy_shape[0].item()).reshape((1, -1, 1)).expand(*shape)
        pys = torch.linspace(1., 0., _grid_xy_shape[1].item()).reshape((-1, 1, 1)).expand(*shape)
        pzs = pzs.reshape((1, 1, -1)).expand(*shape)

        return torch.stack([pxs, pys, pzs], dim=3)

    @staticmethod
    def _linear_blend_patch_weight(grid_shape_2d, half_blend_percent):
        assert 0 <= half_blend_percent[0] <= 0.5, "Blend percent X should be between 0 and 0.5."
        assert 0 <= half_blend_percent[1] <= 0.5, "Blend percent Y should be between 0 and 0.5."

        MIN_WEIGHT = 1e-3
        weight_tensor_x = torch.ones(grid_shape_2d, dtype=torch.float64)
        weight_tensor_y = torch.ones(grid_shape_2d, dtype=torch.float64)

        idx_x = math.floor(grid_shape_2d[0] * half_blend_percent[0])
        idx_y = math.floor(grid_shape_2d[1] * half_blend_percent[1])

        if idx_x > 0:
            weight_tensor_x[:, :idx_x] = torch.linspace(MIN_WEIGHT, 1, idx_x, dtype=torch.float64).unsqueeze(0).expand(
                grid_shape_2d[0], idx_x
            )
            weight_tensor_x[:, -idx_x:] = torch.linspace(1, MIN_WEIGHT, idx_x, dtype=torch.float64).unsqueeze(0).expand(
                grid_shape_2d[0], idx_x
            )

        if idx_y > 0:
            weight_tensor_y[:idx_y, :] = torch.linspace(MIN_WEIGHT, 1, idx_y, dtype=torch.float64).unsqueeze(1).expand(
                idx_y, grid_shape_2d[1]
            )
            weight_tensor_y[-idx_y:, :] = torch.linspace(1, MIN_WEIGHT, idx_y, dtype=torch.float64).unsqueeze(1).expand(
                idx_y, grid_shape_2d[1]
            )

        return weight_tensor_x * weight_tensor_y

    def generate_dsm(self, save_to: str):
        device = self.device
        patch_weight = self.patch_weight.detach().to(device)
        tiff_data = RasterData()

        tiff_data.set_transform(
            bl_bound=[self.l_bound, self.b_bound],
            tr_bound=[self.r_bound, self.t_bound],
            pixel_size=self.pixel_size,
            crs_epsg=self.crs_epsg
        )

        dsm_tensor = torch.zeros(self.dsm_shape, dtype=torch.float64).to(device)
        weight_tensor = torch.zeros(self.dsm_shape, dtype=torch.float64).to(device)

        start_time = time.time()

        for vis_data in tqdm(self.data_loader, desc="Generating DSM"):
            if not vis_data['is_valid'][0]:
                continue

            min_bound = vis_data['min_bound'].squeeze().double()
            max_bound = vis_data['max_bound'].squeeze().double()

            min_bound_center = min_bound + self.pixel_size / 2.
            max_bound_center = max_bound - self.pixel_size / 2.

            self.model.eval()
            with torch.no_grad():
                input_cloud = vis_data.get('inputs').to(device) if self.use_cloud else None
                input_image = vis_data.get('image').to(device) if self.use_image else None

                h_grid = self.model(input_cloud=input_cloud, input_image=input_image)[0].flip(1).squeeze()

            l_col, b_row = tiff_data.query_col_row(min_bound_center[0].item(), min_bound_center[1].item())
            r_col, t_row = tiff_data.query_col_row(max_bound_center[0].item(), max_bound_center[1].item())

            weighted_h_grid = h_grid * patch_weight
            dsm_tensor[t_row:b_row + 1, l_col:r_col + 1] += weighted_h_grid
            weight_tensor[t_row:b_row + 1, l_col:r_col + 1] += patch_weight

        dsm_tensor /= weight_tensor
        dsm_tensor = torch.maximum(dsm_tensor, torch.tensor(0.))

        logging.info(f"DSM Generation time: {time.time() - start_time:.2f} seconds.")

        tiff_data.set_data(dsm_tensor, 1)
        tiff_writer = RasterWriter(tiff_data)
        tiff_writer.write_to_file(save_to)

        return tiff_writer
