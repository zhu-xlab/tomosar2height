from collections import defaultdict
from typing import Dict

import numpy as np
from rasterio.transform import Affine
from tabulate import tabulate
from datetime import datetime

from utils.dilate_mask import dilate_mask
from utils.io_raster import RasterReader


class DSMEvaluator:
    def __init__(self, gt_dsm_path: str, gt_mask_path: str = None, other_mask_path_dict: Dict[str, str] = None):
        self._gt_dsm_reader = RasterReader(gt_dsm_path)
        self.gt_dsm = self._gt_dsm_reader.get_data()

        self.gt_mask = (
            RasterReader(gt_mask_path).get_data().astype(bool)
            if gt_mask_path is not None else np.ones(self.gt_dsm.shape, dtype=bool)
        )

        self.other_mask = None
        if other_mask_path_dict:
            self.other_mask = {
                key: RasterReader(path).get_data().astype(bool)
                for key, path in other_mask_path_dict.items()
            }
            if 'building' in self.other_mask:
                self.other_mask['building'] = dilate_mask(self.other_mask['building'], iterations=2)
                self.other_mask['terrain'] = ~self.other_mask['building']

    def eval(self, target_dsm: np.ndarray, T: Affine):
        target_shape = target_dsm.shape
        tl_bound = T * np.array([0, 0])
        l_col, t_row = np.floor(self._gt_dsm_reader.T_inv * tl_bound).astype(int)

        gt_dsm_clip = self.gt_dsm[t_row:t_row + target_shape[0], l_col:l_col + target_shape[1]]
        gt_mask_clip = self.gt_mask[t_row:t_row + target_shape[0], l_col:l_col + target_shape[1]]

        residuals = target_dsm - gt_dsm_clip
        residuals_masked = residuals[gt_mask_clip]
        residuals_masked = residuals_masked[~np.isnan(residuals_masked)]

        output_stats = defaultdict()
        output_stats['overall'] = self.calculate_statistics(residuals_masked)

        if self.other_mask:
            for land_type, mask in self.other_mask.items():
                mask_clip = mask[t_row:t_row + target_shape[0], l_col:l_col + target_shape[1]]
                gt_land_mask = gt_mask_clip & mask_clip

                land_residuals = residuals[gt_land_mask]
                land_residuals = land_residuals[~np.isnan(land_residuals)]
                output_stats[land_type] = self.calculate_statistics(land_residuals)

        diff_arr = residuals * gt_mask_clip
        diff_arr[~gt_mask_clip] = np.nan

        return output_stats, diff_arr

    @staticmethod
    def calculate_statistics(residual: np.ndarray):
        if residual.size > 0:
            residual_abs = np.abs(residual)
            return {
                'max': np.max(residual),
                'min': np.min(residual),
                'MAE': np.mean(residual_abs),
                'RMSE': np.sqrt(np.mean(residual ** 2)),
                'abs_median': np.median(residual_abs),
                'median': np.median(residual),
                'n_pixel': residual.size,
                'NMAD': 1.4826 * np.median(np.abs(residual - np.median(residual)))
            }
        return {
            'max': None, 'min': None, 'MAE': None, 'RMSE': None,
            'abs_median': None, 'median': None, 'n_pixel': None, 'NMAD': None
        }


def print_statistics(statistics: Dict, title: str, save_to: str = None):
    metrics = {
        'MAE[m]': 'MAE', 'RMSE[m]': 'RMSE', 'MedAE[m]': 'abs_median',
        'Max[m]': 'max', 'Min[m]': 'min', 'Median[m]': 'median',
        'NMAD[m]': 'NMAD', '#Pixels': 'n_pixel'
    }

    header = list(metrics.keys())
    content = []
    for land_type, stats in statistics.items():
        content.append([land_type.capitalize()] + [stats[metrics[m]] for m in header])

    header.insert(0, 'Type')
    output = [
        "DSM Evaluation\t\t\tcreated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        title,
        "Performance Evaluation", "=" * 20,
        tabulate(content, headers=header, tablefmt="simple", floatfmt=".4f"),
        "-" * 20,
        """ Metrics:
        MAE: Mean Absolute residual Error
        RMSE: Root Mean Square Error
        MedAE: Median Absolute Error
        Max: Maximum value
        Min: Minimum value
        Median: Median value
        NMAD: Normalised Median Absolute Deviation
        #pixels: Number of pixels
        """
    ]

    result = '\n'.join(output)
    if save_to:
        with open(save_to, 'w+') as f:
            f.write(result)

    return result


if __name__ == '__main__':
    output_tiff = "/workspace/projects/tomosar2height/data/berlin/raster/berlin_ndsm_bilinear.tif"
    ground_truth_tiff = "/workspace/projects/tomosar2height/data/berlin/raster/berlin_ndsm_gt.tif"
    mask_paths = {
        'building': '/workspace/projects/tomosar2height/data/berlin/raster/footprint_rotate12deg_aoi.tif'
    }

    output_reader = RasterReader(output_tiff)
    evaluator = DSMEvaluator(ground_truth_tiff, other_mask_path_dict=mask_paths)
    stats, residuals = evaluator.eval(output_reader.get_data(), output_reader.T)

    print(print_statistics(stats, title="DSM Evaluation"))
