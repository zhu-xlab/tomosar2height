import logging
import math
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine


class RasterData:
    def __init__(self):
        """Initialize an empty raster data object."""
        self._editable = True
        self._data: Dict = defaultdict()
        self._n_rows: int = None
        self._n_cols: int = None
        self.T: Affine = None
        self.T_inv: Affine = None
        self.pixel_size: List[float] = None
        self.crs: rasterio.crs.CRS = None
        self.tiff_file: str = None

    def get_data(self, band=1) -> np.ndarray:
        """Retrieve data for a specific band."""
        out = self._data.get(band, None)
        if out is not None:
            out = out.copy()
        return out

    def set_data(self, data, band=1):
        """Set data for a specific band."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self._is_shape_consistent({band: data}):
            self._data[band] = data
            self._n_rows, self._n_cols = data.shape
        else:
            logging.warning("Cannot set data: Data shape not consistent.")

    def _is_shape_consistent(self, data_dict: dict) -> bool:
        """Check if the data shapes are consistent across bands."""
        _n_rows = self._n_rows
        _n_cols = self._n_cols
        for _, v in data_dict.items():
            height, width = v.shape
            if _n_rows is None or _n_cols is None:
                _n_rows = height
                _n_cols = width
            elif (_n_rows != height) or (_n_cols != width):
                return False
        return True

    def set_transform(self, bl_bound, tr_bound, pixel_size, crs_epsg):
        """Set the transformation parameters for the raster data."""
        if self._editable:
            self.pixel_size = np.array(pixel_size).tolist()
            self.T = Affine(self.pixel_size[0], 0.0, bl_bound[0],
                            0.0, -self.pixel_size[1], tr_bound[1])
            self.T_inv = ~self.T
            self.crs = rasterio.crs.CRS.from_epsg(crs_epsg)
        else:
            logging.warning("Cannot edit this RasterData.")

    def set_transform_from(self, target_data):
        """Set the transformation parameters from another raster data object."""
        if self._editable:
            self.pixel_size = target_data.pixel_size
            self.T = target_data.T
            self.T_inv = target_data.T_inv
            self.crs = target_data.crs
        else:
            logging.warning("Cannot edit this RasterData.")

    @staticmethod
    def cal_dsm_shape(bl_bound, tr_bound, pixel_size):
        """
        Calculate DSM raster dimensions based on the bounding box.

        Args:
            bl_bound: Bottom-left bounding point.
            tr_bound: Top-right bounding point.
            pixel_size: DSM pixel size.

        Returns:
            Tuple[int, int]: Number of rows and columns.
        """
        bl_bound = np.array(bl_bound).astype(np.float64)
        tr_bound = np.array(tr_bound).astype(np.float64)
        pixel_size = np.array(pixel_size).astype(np.float64)
        _n_rows = math.floor((tr_bound[1] - bl_bound[1]) / pixel_size[1])
        _n_cols = math.floor((tr_bound[0] - bl_bound[0]) / pixel_size[0])
        return _n_rows, _n_cols

    def is_complete(self) -> bool:
        """Check if the raster data object is complete."""
        return (len(self._data) > 0 and
                self._is_shape_consistent(self._data) and
                self._n_rows is not None and
                self._n_cols is not None and
                self.T is not None and
                self.T_inv is not None and
                self.crs is not None)

    def query_value(self, x, y, band=1):
        """Query a value at a specific coordinate."""
        col, row = self.query_col_row(x, y)
        if self.is_in(col, row, band):
            return self._data[band][row, col]
        return None

    def is_in(self, col, row, band) -> Union[bool, np.ndarray]:
        """Check if a coordinate is within the raster bounds."""
        shape = self._data[band].shape
        if isinstance(col, (int, np.integer)) and isinstance(row, (int, np.integer)):
            return 0 <= row < shape[0] and 0 <= col < shape[1]
        elif isinstance(col, np.ndarray) and isinstance(row, np.ndarray):
            return np.where(((0 <= row) & (row < shape[0]) & (0 <= col) & (col < shape[1])), True, False)
        raise TypeError("col and row should both be int or np.ndarray.")

    def query_col_row(self, x, y):
        """Convert geographic coordinates to raster indices."""
        cols, rows = self.query_col_rows(np.array([[x, y]]))
        return cols[0], rows[0]

    def query_col_rows(self, xy_arr: np.ndarray):
        """Convert multiple geographic coordinates to raster indices."""
        cols, rows = np.floor(self.T_inv * xy_arr.T).astype(int)
        return cols, rows

    def query_values(self, xy_arr: np.ndarray, band=1, outer_value=-99999):
        """Query multiple values at specific coordinates."""
        cols, rows = self.query_col_rows(xy_arr)
        tiff_data = self._data[band]
        is_in = self.is_in(cols, rows, band)
        rows = rows[is_in]
        cols = cols[is_in]
        pixels = np.full(xy_arr.shape[0], outer_value, dtype=tiff_data.dtype)
        pixels[is_in] = np.array([tiff_data[rows[i], cols[i]] for i in range(len(rows))])
        return pixels

    def query_value_3d_points(self, points, band=1, outer_value=0):
        """Query values for 3D points."""
        if points.shape[0] == 0:
            return np.empty(0)
        xy_arr = points[:, :2]
        return self.query_values(xy_arr, band, outer_value)


class RasterReader(RasterData):
    def __init__(self, tiff_file):
        """Initialize a raster reader for a given TIFF file."""
        super().__init__()
        self.tiff_file = tiff_file
        self.dataset_reader: rasterio.DatasetReader = rasterio.open(tiff_file)
        self.from_reader(self.dataset_reader)

    def from_reader(self, tiff_obj: rasterio.DatasetReader):
        """Load raster data from a Rasterio dataset reader."""
        if self._editable:
            self._data = {i: tiff_obj.read(i) for i in range(1, tiff_obj.count + 1)}
            self.T = tiff_obj.transform
            self.T_inv = ~self.T
            self.pixel_size = [self.T.a, -self.T.e]
            self.crs = self.dataset_reader.crs
            self._editable = False
        else:
            logging.warning("Cannot edit this RasterData (from_reader called).")


class RasterWriter(RasterData):
    dataset_writer: rasterio.io.DatasetWriter

    def __init__(self, raster_data: RasterData, dtypes=('float32')):
        """Initialize a raster writer from existing raster data."""
        super().__init__()
        super().__dict__.update(raster_data.__dict__)
        self.dtypes = dtypes

    def write_to_file(self, filename: str) -> bool:
        """Write the raster data to a file."""
        if self.is_complete():
            n_channel = len(self._data)
            self.tiff_file = filename
            self._open_file(filename)
            for c in range(1, n_channel + 1):
                self.dataset_writer.write(self._data[c].astype(np.float32), c)
            self._close_file()
            return True
        else:
            logging.warning("RasterData is not complete, cannot write to TIFF file.")
            return False

    def _open_file(self, filename):
        """Open a file for writing."""
        self.dataset_writer = rasterio.open(
            filename,
            'w+',
            driver='GTiff',
            height=self._n_rows,
            width=self._n_cols,
            count=len(self._data),
            dtype=self.dtypes,
            crs=self.crs,
            transform=self.T
        )

    def _close_file(self):
        """Close the open file."""
        self.dataset_writer.close()
