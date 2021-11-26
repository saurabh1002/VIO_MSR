#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import os
import pickle
import argparse

import cv2
import open3d as o3d
import scipy.spatial
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
sys.path.append('..')

from utils.utils import wrapTo2Pi


def process_input_data(bboxes_path, associations_path):
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

def get_depth_mask(depth_frame_shape, keypoint):
    depth_mask = np.zeros(depth_frame_shape, np.uint8)
    depth_mask[int(keypoint[1]), int(keypoint[0])] = 1
    return depth_mask

def find_k_nearest(keypoints: np.ndarray, k: int = 4):
    dist = scipy.spatial.distance.cdist(keypoints, keypoints)
    sort_ids = np.argsort(dist)

    return sort_ids[:, 1:k+1]

def compute_descriptor(keypoint: np.ndarray, k_neigbors: np.ndarray):
    k = k_neigbors.shape[0]
    dist = la.norm(keypoint - k_neigbors, 2, 1)
    
    dist_ratios = dist[:-1] / dist[-1]
    angles = np.arctan2(k_neigbors[:-1, 1] - keypoint[1], k_neigbors[:-1, 0] - keypoint[0]) - np.arctan2(k_neigbors[-1, 1] - keypoint[1], k_neigbors[-1, 0] - keypoint[0]) 
    angles = np.array([wrapTo2Pi(-angle) for angle in angles]) / (2 * np.pi)

    sort_idx = np.argsort(angles)

    desc = np.r_[keypoint, dist_ratios[sort_idx], angles[sort_idx]]
    return desc, sort_idx


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

    parser.add_argument('--k', default=4, type=int, help='Number of nearest neighbors to use for descriptor')
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

        rgb_frame_2 = cv2.imread(args.data_root_path + rgb_names[n + 1])
        depth_frame_2 = cv2.imread(args.data_root_path + depth_names[n + 1])
        keypts_2d_2 = np.array(bboxes_d[os.path.basename(rgb_names[n + 1])])[:, 4:]

        keypts_3d_1 = []
        for i in range(keypts_2d_1.shape[0]):
            depth_mask = get_depth_mask(depth_frame_1.shape, keypts_2d_1[i])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_frame_1), 
                o3d.geometry.Image(np.multiply(depth_frame_1, depth_mask)), 
                depth_scale=1000, depth_trunc=50, convert_rgb_to_intensity=True)
            pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, rgb_camera_intrinsic)
            print(pcl.points)
            keypts_3d_1.append(list(pcl.points))

        keypts_3d_2 = []
        for i in range(keypts_2d_2.shape[0]):
            depth_mask = get_depth_mask(depth_frame_2.shape, keypts_2d_2[i])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_frame_2), 
                o3d.geometry.Image(np.multiply(depth_frame_2, depth_mask)), 
                depth_scale=1, depth_trunc=50, convert_rgb_to_intensity=True)
            pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, rgb_camera_intrinsic)
            keypts_3d_2.append(np.array(pcl.points) * 100)


        # print(np.array(keypts_3d_1))
        

        if args.visualize:
            o3d.visualization.draw_geometries([pcl_1], 'Keypoints frame {}'.format(n + 1))
            o3d.visualization.draw_geometries([pcl_2], 'Keypoints frame {}'.format(n + 2))


        