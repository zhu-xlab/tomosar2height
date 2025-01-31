from typing import Union, Tuple

import numpy as np
import open3d as o3d
import torch


def crop_pc_2d_index(points: Union[np.ndarray, torch.Tensor], p_min, p_max):
    """
    Find indices of points within a 2D bounding box.

    Args:
        points (Union[np.ndarray, torch.Tensor]): Input points.
        p_min (array-like): Bottom-left corner of the bounding box.
        p_max (array-like): Top-right corner of the bounding box.

    Returns:
        Union[np.ndarray, torch.Tensor]: Indices of points within the bounding box.
    """
    if isinstance(points, np.ndarray):
        index = np.where((points[:, 0] > p_min[0]) & (points[:, 0] < p_max[0]) &
                         (points[:, 1] > p_min[1]) & (points[:, 1] < p_max[1]))[0]
        return index
    elif isinstance(points, torch.Tensor):
        index = torch.where((points[:, 0] > p_min[0]) & (points[:, 0] < p_max[0]) &
                            (points[:, 1] > p_min[1]) & (points[:, 1] < p_max[1]))[0]
        return index
    else:
        raise NotImplementedError("Unsupported type for points.")


def crop_pc_2d(points: Union[np.ndarray, torch.Tensor], p_min, p_max) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Crop a 2D point cloud based on an x-y bounding box.

    Args:
        points (Union[np.ndarray, torch.Tensor]): Input points.
        p_min (array-like): Bottom-left corner of the bounding box.
        p_max (array-like): Top-right corner of the bounding box.

    Returns:
        Tuple: Cropped points and their indices.
    """
    if isinstance(points, (np.ndarray, torch.Tensor)):
        index = crop_pc_2d_index(points, p_min, p_max)
        new_points = points[index]
        return new_points, index
    else:
        raise NotImplementedError("Unsupported type for points.")


def crop_pc_3d(points: Union[np.ndarray, o3d.geometry.PointCloud], p_min, p_max) -> Tuple[
    np.ndarray, o3d.geometry.PointCloud]:
    """
    Crop a 3D point cloud using a 3D axis-aligned bounding box.

    Args:
        points (Union[np.ndarray, o3d.geometry.PointCloud]): Input points or Open3D point cloud.
        p_min (array-like): Minimum corner of the bounding box.
        p_max (array-like): Maximum corner of the bounding box.

    Returns:
        Tuple: Cropped points as a NumPy array and the cropped Open3D point cloud.
    """
    if isinstance(points, o3d.geometry.PointCloud):
        pcd = points
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

    bbox = o3d.geometry.AxisAlignedBoundingBox(p_min, p_max)
    cropped_pcd = pcd.crop(bbox)

    return np.asarray(cropped_pcd.points).copy(), cropped_pcd
