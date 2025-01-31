"""
Coordinate transform utilities.
"""

from typing import Union

import numpy as np
import open3d as o3d
import torch


def coordinate2index(x, reso, coord_type='2d'):
    """
    Generate grid index of points.

    Args:
        x (tensor): Points normalized to [0, 1].
        reso (int): Grid resolution.
        coord_type (str): Coordinate type ('2d').

    Returns:
        tensor: Grid indices.
    """
    x = (x * reso).long()
    if coord_type == '2d':
        index = x[:, :, 0] + reso * x[:, :, 1]
    index = index[:, None, :]
    return index


def normalize_3d_coordinate(p, padding=0):
    """
    Normalize 3D coordinates to [0, 1] for unit cube experiments.

    Args:
        p (tensor): Points.
        padding (float): Padding parameter, so [-0.5, 0.5] -> [-0.55, 0.55].

    Returns:
        tensor: Normalized coordinates.
    """
    raise NotImplementedError("This function is not implemented yet.")


def make_3d_grid(bb_min, bb_max, shape):
    """
    Create a 3D grid.

    Args:
        bb_min (tuple): Bounding box minimum.
        bb_max (tuple): Bounding box maximum.
        shape (tuple): Output shape.

    Returns:
        tensor: 3D grid points.
    """
    size = shape[0] * shape[1] * shape[2]
    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])
    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)
    return p


def normalize_pc(points: Union[np.ndarray, o3d.geometry.PointCloud], scales, center_shift):
    """
    Normalize a point cloud.

    Args:
        points: Input point cloud.
        scales: Scale factor.
        center_shift: Shift for center alignment.

    Returns:
        np.ndarray: Normalized point cloud.
    """
    if isinstance(points, o3d.geometry.PointCloud):
        points = np.asarray(points.points)
    return (points - center_shift) / scales


def invert_normalize_pc(points: Union[np.ndarray, o3d.geometry.PointCloud], scales, center_shift):
    """
    Invert normalization of a point cloud.

    Args:
        points: Input point cloud.
        scales: Scale factor.
        center_shift: Shift for center alignment.

    Returns:
        np.ndarray: Original point cloud.
    """
    if isinstance(points, o3d.geometry.PointCloud):
        points = np.asarray(points.points)
    return points * scales + center_shift


def apply_transform(p, M):
    """
    Apply a transformation matrix to points.

    Args:
        p: Points.
        M: Transformation matrix.

    Returns:
        Transformed points.
    """
    if isinstance(p, np.ndarray):
        p = np.hstack([p, np.ones((p.shape[0], 1))]).T
        p2 = np.matmul(M, p).T
        return (p2[:, :3] / p2[:, 3:4])
    elif isinstance(p, torch.Tensor):
        p = torch.cat([p, torch.ones((p.shape[0], 1), device=p.device)], dim=1).T
        p2 = torch.matmul(M, p).T
        return p2[:, :3] / p2[:, 3:4]
    else:
        raise TypeError("Invalid type for points.")


def invert_transform(M):
    """
    Invert a transformation matrix.

    Args:
        M: Transformation matrix.

    Returns:
        Inverted matrix.
    """
    if isinstance(M, np.ndarray):
        return np.linalg.inv(M)
    elif isinstance(M, torch.Tensor):
        return torch.inverse(M)
    else:
        raise TypeError("Invalid type for matrix.")


def stack_transforms(M_ls):
    """
    Stack multiple transformation matrices.

    Args:
        M_ls (list): List of transformation matrices.

    Returns:
        Combined transformation matrix.
    """
    M_out = M_ls[0]
    for M in M_ls[1:]:
        M_out = M_out @ M
    return M_out
