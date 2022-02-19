#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# This script defines some utility functions that are used often in other scripts

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
from typing import NoReturn
import open3d as o3d


def read_file(path: str, delimiter: str = ' ') -> (np.ndarray):
    """ Read data from a file

    Arguments
    ---------
    - path: Path of ASCII file to read
    - delimiter: Delimiter character for the file to be read

    Returns
    -------
    - data: Data from file as a numpy array
    """
    data = np.loadtxt(path, delimiter=delimiter)

    return data


def wrapTo2Pi(theta: float) -> (float):
    """
    Wrap around angles to the range [0, 2pi]
    
    Arguments
    ---------
    - theta: angle
    Returns
    -------
    - theta: angle within range [0, 2pi]
    """
    while theta < 0:
        theta += 2 * np.pi
    while theta > 2 * np.pi:
        theta -= 2 * np.pi
    return theta

def toHomogenous(points: np.ndarray):
    return np.hstack((points, np.ones((points.shape[0], 1))))

def toEuclidean(points: np.ndarray):
    return (points / points[:, -1].reshape(-1, 1))[:, :-1]
#Create RGBD Image
#Create PointCLoud

def create_rgbdimg(rgb_img: np.ndarray, depth_img: np.ndarray,depth_scale = 1000, depth_trunc= 10.0, convert_rgb_to_intensity = False):

    rgb_o3d = o3d.geometry.Image(rgb_img)
    depth_o3d = o3d.geometry.Image(depth_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=convert_rgb_to_intensity,
        )

    return rgbd

def RGBD2PCL(rgbd_img: o3d.geometry.RGBDImage, camera_intrinsics: np.ndarray,normals: bool):
    I = np.eye(4)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img, camera_intrinsics, I)

    if (normals):
        pcd.estimate_normals()

    return pcd