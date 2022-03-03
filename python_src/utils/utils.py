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
import cv2
import numpy as np
import open3d as o3d
import scipy.spatial.transform as tf

from typing import NoReturn


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


def se3_to_SE3(rvec: np.ndarray, tvec: np.ndarray) -> (np.ndarray):
    T = np.eye(4)
    T[:-1,:-1] = cv2.Rodrigues(rvec)[0]
    T[:-1,-1] = tvec.reshape((3,))
    return T


def odom_from_SE3(t: float, TF: np.ndarray, change_basis: np.ndarray = np.eye(3)) -> (list):
    origin = change_basis @ TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:-1, :-1]).as_quat()
    return list(np.r_[t, origin, rot_quat])


def create_rgbdimg(rgb_img: np.ndarray,
                   depth_img: np.ndarray,
                   depth_scale: float = 1000,
                   depth_trunc: float = 5.0,
                   convert_rgb_to_intensity: bool = False,
                   tensor: bool = False,
                   device: str = o3d.core.Device("CUDA:0")
                  ):
    if not tensor:
        rgb_o3d = o3d.geometry.Image(rgb_img)
        depth_o3d = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale = depth_scale,
                depth_trunc = depth_trunc,
                convert_rgb_to_intensity = convert_rgb_to_intensity,
            )
    else:
        rgb_o3d = o3d.t.geometry.Image(rgb_img)
        depth_o3d = o3d.t.geometry.Image(depth_img)
        rgbd = o3d.t.geometry.RGBDImage(rgb_o3d, depth_o3d).to(device)

    return rgbd


def RGBD2PCL(rgbd_img: o3d.geometry.RGBDImage,
            camera_intrinsics,
            compute_normals: bool, 
            tensor: bool = False,
            depth_scale: float = 1000,
            depth_trunc: float = 5.0,
            device: str = o3d.core.Device("CUDA:0")
            ):
    global_pose = np.eye(4)
    if not tensor:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_img, camera_intrinsics, global_pose)
        if (compute_normals):
            pcd.estimate_normals()
    else:
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img,
            o3d.core.Tensor(camera_intrinsics.intrinsic_matrix, o3d.core.Dtype.Float64),
            o3d.core.Tensor(global_pose, o3d.core.Dtype.Float64),
            depth_scale = depth_scale,
            depth_max = depth_trunc,
            with_normals = compute_normals
        ).to(device)

    return pcd
