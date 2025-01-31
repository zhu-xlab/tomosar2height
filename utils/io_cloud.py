from typing import Union

import laspy
import numpy as np
import open3d as o3d


def load_pc(pc_path: str) -> np.ndarray:
    """
    Load a point cloud from various file formats.

    Args:
        pc_path (str): Path to the point cloud file.

    Returns:
        np.ndarray: Loaded point cloud as a numpy array.
    """
    extension = pc_path.split('.')[-1].lower()
    points: np.ndarray

    if extension == 'las':
        points = load_las_as_numpy(pc_path)
    elif extension == 'npy':
        points = np.load(pc_path)
    elif extension in ['xyz', 'ply', 'pcd', 'pts', 'xyzn', 'xyzrgb']:
        pcd = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd.points)
    else:
        raise TypeError(f"Unknown file type: {extension}")

    return points


def load_las_as_numpy(las_path: str) -> np.ndarray:
    """
    Load a .las point cloud and convert it into a numpy array.

    Args:
        las_path (str): Full path to the .las file.

    Returns:
        np.ndarray: Point cloud as a numpy array.
    """
    with laspy.open(las_path) as f:
        _las = f.read()
    x = np.array(_las.x).reshape((-1, 1))
    y = np.array(_las.y).reshape((-1, 1))
    z = np.array(_las.z).reshape((-1, 1))
    points = np.concatenate([x, y, z], axis=1)
    return points


def save_pc_to_ply(pc_path: str, points: Union[np.ndarray, o3d.geometry.PointCloud], colors: np.ndarray = None):
    """
    Save a point cloud to a .ply file.

    Args:
        pc_path (str): Path to save the .ply file.
        points (Union[np.ndarray, o3d.geometry.PointCloud]): Point cloud data.
        colors (np.ndarray, optional): Point cloud colors.
    """
    if isinstance(points, o3d.geometry.PointCloud):
        pcd = points
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pc_path = pc_path if pc_path.lower().endswith(".ply") else pc_path + ".ply"
    o3d.io.write_point_cloud(pc_path, pcd)
