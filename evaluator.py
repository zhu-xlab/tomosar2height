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
        self.has_binary_building = False
        self.has_ternary_building = False
        
        if other_mask_path_dict:
            self.other_mask = {}
            
            if 'building' in other_mask_path_dict:
                building_mask = RasterReader(other_mask_path_dict['building']).get_data().astype(bool)
                self.other_mask['building'] = dilate_mask(building_mask, iterations=2)
                self.other_mask['terrain'] = ~self.other_mask['building']
                self.has_binary_building = True
            
            if 'type' in other_mask_path_dict:
                type_mask = RasterReader(other_mask_path_dict['type']).get_data()
                
                self.other_mask['non_building'] = (type_mask == 0)
                self.other_mask['residential'] = (type_mask == 1) 
                self.other_mask['non_residential'] = (type_mask == 2)
                self.other_mask['building_combined'] = (type_mask > 0)
                
                self.other_mask['residential'] = dilate_mask(self.other_mask['residential'], iterations=2)
                self.other_mask['non_residential'] = dilate_mask(self.other_mask['non_residential'], iterations=2)
                self.other_mask['building_combined'] = dilate_mask(self.other_mask['building_combined'], iterations=2)
                self.has_ternary_building = True
            
            for key, path in other_mask_path_dict.items():
                if key not in ['building', 'type']:
                    self.other_mask[key] = RasterReader(path).get_data().astype(bool)

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


def print_statistics(statistics: Dict, title: str, save_to: str = None, has_binary: bool = False, has_ternary: bool = False):
    """Print statistics with sections for both binary and ternary building evaluation."""
    metrics = {
        'MAE[m]': 'MAE', 'RMSE[m]': 'RMSE', 'MedAE[m]': 'abs_median',
        'Max[m]': 'max', 'Min[m]': 'min', 'Median[m]': 'median',
        'NMAD[m]': 'NMAD', '#Pixels': 'n_pixel'
    }

    header = list(metrics.keys())
    header.insert(0, 'Type')
    
    output = [
        "DSM Evaluation\t\t\tcreated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        title,
        "Performance Evaluation", "=" * 30
    ]
    
    # Binary Building Section: {overall, terrain, building}
    if has_binary:
        binary_content = []
        binary_keys = ['overall', 'terrain', 'building']
        binary_display = {'overall': 'Overall', 'terrain': 'Terrain', 'building': 'Building'}
        
        for key in binary_keys:
            if key in statistics:
                display_name = binary_display[key]
                binary_content.append([display_name] + [statistics[key][metrics[m]] for m in header[1:]])
        
        if binary_content:
            output.extend([
                "",
                "Binary Building Classification:",
                tabulate(binary_content, headers=header, tablefmt="simple", floatfmt=".4f")
            ])
    
    # Ternary Building Section: {residential, non_residential}
    if has_ternary:
        ternary_content = []
        ternary_keys = ['residential', 'non_residential']
        ternary_display = {'residential': 'Residential', 'non_residential': 'Non Residential'}
        
        for key in ternary_keys:
            if key in statistics:
                display_name = ternary_display[key]
                ternary_content.append([display_name] + [statistics[key][metrics[m]] for m in header[1:]])
        
        if ternary_content:
            output.extend([
                "",
                "Building Type Classification:",
                tabulate(ternary_content, headers=header, tablefmt="simple", floatfmt=".4f")
            ])
    
    # Other masks (if any exist beyond the main categories)
    other_content = []
    processed_keys = {'overall', 'building', 'terrain', 'residential', 'non_residential', 'non_building', 'building_combined'}
    
    for key, stats in statistics.items():
        if key not in processed_keys:
            display_name = key.replace('_', ' ').title()
            other_content.append([display_name] + [stats[metrics[m]] for m in header[1:]])
    
    if other_content:
        output.extend([
            "",
            "Other Classifications:",
            tabulate(other_content, headers=header, tablefmt="simple", floatfmt=".4f")
        ])
    
    output.extend([
        "",
        "-" * 30,
        """ Metrics:
        MAE: Mean Absolute residual Error
        RMSE: Root Mean Square Error
        MedAE: Median Absolute Error
        Max: Maximum value
        Min: Minimum value
        Median: Median value
        NMAD: Normalised Median Absolute Deviation
        #pixels: Number of pixels
        
        Binary Building Classes:
        - Overall: All valid pixels
        - Terrain: Non-building areas (from binary mask)
        - Building: Building areas (from binary mask)
        
        Building Type Classes:
        - Residential: Residential buildings (class 1)
        - Non Residential: Non-residential buildings (class 2)
        """
    ])

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

    print(print_statistics(stats, title="DSM Evaluation", has_binary=True, has_ternary='type' in mask_paths))
