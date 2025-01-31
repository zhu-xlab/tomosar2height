"""
Read TomoSAR point clouds from FLT file.
"""

import struct
import numpy as np


def readFLT1D(filename):
    """
    Read one-dimension point records from FLT file.
    """
    fid = open(filename, "rb")
    header_str = fid.read(4 * 8)
    header_be = struct.unpack("8i", header_str)

    if header_be[0] != 1504078485:
        header_le = struct.unpack('>8i', header_str)
        header = header_le
    else:
        header = header_be

    data = np.zeros(header[1] * header[2], dtype=np.float32)

    for id in range(0, header[1] * header[2] - 1):
        data_str = fid.read(4)
        data[id], = struct.unpack("f", data_str)

    fid.close()

    data = data.reshape(header[2], header[1])

    return header, data


def readFLT3D():
    _, data_x = readFLT1D('../data/ps_x.utm.shifted.flt')
    _, data_y = readFLT1D('../data/ps_y.utm.shifted.flt')
    _, data_z = readFLT1D('../data/ps_height.utm.shifted.flt')

    data = np.concatenate([data_x, data_y, data_z], axis=0).transpose()
    print('data.shape:', data.shape)

    np.save('../data/ps_xyh.utm.shifted.xyz', data)


if __name__ == '__main__':
    readFLT3D()
