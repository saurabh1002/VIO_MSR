#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import pickle
import argparse

import cv2
import open3d as o3d
import scipy.spatial
import numpy as np
import numpy.linalg as la

from tqdm import tqdm

import os
import sys
sys.path.append('..')

from utils.ransac_homography import *
from utils.utils import wrapTo2Pi

def process_input_data(bboxes_path: str, associations_path: str):
    with open(bboxes_path, "rb") as f:
        bboxes_data = pickle.load(f)
    
    with open(associations_path, 'r') as f:
        depth_frame_names = []
        rgb_frame_names = []

        for line in f.readlines():
            _, rgb_path, _, depth_path = line.rstrip("\n").split(' ')
            if(os.path.basename(rgb_path) in list(bboxes_data.keys())):
                depth_frame_names.append(depth_path)
                rgb_frame_names.append(rgb_path)

    return bboxes_data, rgb_frame_names, depth_frame_names

def get_depth_mask(depth_frame_shape: tuple, keypoint: np.ndarray) -> (np.ndarray):
    '''This function get the depth mask for a pixel

    Arguments
    ---------
    - depth_frame_shape: shape of the depth frame
    - keypoint: keypoint of interest

    Returns
    -------
    - depth_mask: depth mask for a pixel
    '''
    depth_mask = np.zeros(depth_frame_shape, np.uint8)
    depth_mask[int(keypoint[1]), int(keypoint[0])] = 1
    return depth_mask


def compute_hist_3d(keypts_3d: np.ndarray, nbins: int = 8) -> (np.ndarray):
    hist_3d = np.zeros((keypts_3d.shape[0], nbins))
    for i, kp in enumerate(keypts_3d):
        rel_kps = np.delete(keypts_3d, i, 0) - kp
        x, y, z = rel_kps[:, 0], rel_kps[:, 1], rel_kps[:, 2]

        hist_3d[i, 0] = np.sum((x >= 0) & (y >= 0) & (z >= 0))
        hist_3d[i, 1] = np.sum((x >= 0) & (y >= 0) & (z < 0))
        hist_3d[i, 2] = np.sum((x >= 0) & (y < 0) & (z >= 0))
        hist_3d[i, 3] = np.sum((x >= 0) & (y < 0) & (z < 0))
        hist_3d[i, 4] = np.sum((x < 0) & (y >= 0) & (z >= 0))
        hist_3d[i, 5] = np.sum((x < 0) & (y >= 0) & (z < 0))
        hist_3d[i, 6] = np.sum((x < 0) & (y < 0) & (z >= 0))
        hist_3d[i, 7] = np.sum((x < 0) & (y < 0) & (z < 0))

    return hist_3d / keypts_3d.shape[0]

def compute_descriptor(keypoints_2d: np.ndarray, hist: np.ndarray):
    descriptors = np.zeros((keypoints_2d.shape[0], 2 + hist.shape[1]))
    for i, kp in enumerate(keypoints_2d):
        descriptors[i] = np.r_[kp, hist[i]]
    return descriptors

def get_keypoints(rgb_frame, depth_frame, keypts_2d):
    keypts_3d = []
    pcl_gen_flag = []

    for keypt in keypts_2d:
        depth_mask = get_depth_mask(depth_frame.shape, keypt)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_frame), 
            o3d.geometry.Image(np.multiply(depth_frame, depth_mask)), 
            depth_scale=1000, convert_rgb_to_intensity=True)
        pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, rgb_camera_intrinsic)
        pcl_gen_flag.append(pcl.has_points())
        if(pcl.has_points()):
            keypts_3d.append(np.array(pcl.points)[0])
    
    keypts_2d = np.array(keypts_2d)[pcl_gen_flag]
    keypts_3d = np.array(keypts_3d)

    return keypts_2d, keypts_3d


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''
    This script generates custom descriptors for 3D macro keypoints
    ''')
    # Dataset paths
    parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/bboxes.pickle', type=str,
        help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations.txt', type=str,
        help='Path to the associations file for RGB and Depth frames')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save_descriptor', default=False, type=bool, help='Save computed Descriptors as .npz file')
    args = parser.parse_args()

    bboxes_d, rgb_names, depth_names = process_input_data(args.bboxes_path, args.associations_path)
     
    num_of_frames = len(rgb_names)
    
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)

    for n in range(num_of_frames - 1):
        rgb_frame_1 = cv2.imread(args.data_root_path + rgb_names[n])
        depth_frame_1 = cv2.imread(args.data_root_path + depth_names[n])
        keypts_2d_1 = np.array(bboxes_d[os.path.basename(rgb_names[n])])[:, 4:]
        keypts_2d_1, keypts_3d_1 = get_keypoints(rgb_frame_1, depth_frame_1, keypts_2d_1)

        rgb_frame_2 = cv2.imread(args.data_root_path + rgb_names[n + 1])
        depth_frame_2 = cv2.imread(args.data_root_path + depth_names[n + 1])
        keypts_2d_2 = np.array(bboxes_d[os.path.basename(rgb_names[n + 1])])[:, 4:]
        keypts_2d_2, keypts_3d_2 = get_keypoints(rgb_frame_2, depth_frame_2, keypts_2d_2)

        hist_1 = compute_hist_3d(keypts_3d_1)
        hist_2 = compute_hist_3d(keypts_3d_2)

        descriptors_1 = compute_descriptor(keypts_2d_1, hist_1)
        descriptors_2 = compute_descriptor(keypts_2d_2, hist_2)

        M = compute_matches(descriptors_1[:, 2:], descriptors_2[:, 2:], rgb_frame_1.shape[0], rgb_frame_1.shape[1])

        P1 = descriptors_1[:, :2].astype(int)
        P2 = descriptors_2[:, :2].astype(int)

        if(M.shape[0] > 0):
            H, M = compute_homography_ransac(P1, P2, M)

        w = rgb_frame_1.shape[1]
        rgb_match_frame = np.concatenate((rgb_frame_1, rgb_frame_2), 1)

        for kp in descriptors_1:
            cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 10, (255, 0, 0), 2)

        for kp in descriptors_2:
            cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                       int(kp[1])), 10, (0, 0, 255), 2)

        for kp_l, kp_r in zip(descriptors_1[M[:, 0]].astype(int), descriptors_2[M[:, 1]].astype(int)):
            cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)

        if args.visualize:
            cv2.imshow('matches', rgb_match_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.save_descriptor:
            cv2.imwrite(
                '../../eval_data/custom_3d_desc/{}.png'.format(n), rgb_match_frame)