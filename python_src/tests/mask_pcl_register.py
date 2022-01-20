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
from tqdm import tqdm

import cv2
import open3d as o3d
from scipy.spatial import transform as tf
import numpy as np

from matplotlib import pyplot as plt

import os
import sys
import copy
sys.path.append(os.pardir)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def process_input_data(bboxes_path: str, associations_path: str) -> tuple[dict, list, list]:
    ''' Loads the input data for further use

    Arguments
    ---------
    - bboxes_path: Path to the pickle file containing the bounding box data for each rgb frame
    - associations_path: Path to the txt file conatining the associations for RGB and Depth frames from the dataset
    
    Returns
    -------
    - bboxes_data: {rgb_frame_path: bboxes} a dictionary containing the bounding boxes in each rgb frame as key-value pairs
    - rgb_frame_names: list containing filenames for all rgb frames from the associations file
    - depth_frame_names: list containing filenames for all depth frames from the associations file
    '''
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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script generates custom descriptors for 3D macro keypoints''')

    # Dataset paths
    parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/detection.pickle', type=str,
        help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations_rgbd.txt', type=str,
        help='Path to the associations file for RGB and Depth frames')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
    args = parser.parse_args()

    bboxes_d, rgb_names, depth_names = process_input_data(args.bboxes_path, args.associations_path)
    num_of_frames = len(rgb_names)

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if args.data_root_path == "../../datasets/phenorob/images_apples_right/":
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif args.data_root_path == '../../datasets/phenorob/images_apples/':
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    

    # [x, y, z, qx, qy, qz, qw]
    SE3_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    T = np.eye(4)

    merged_pcd = o3d.geometry.PointCloud()

    bbox_size = 10
    skip_frames = 15
    for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):
        # n th frame
        rgb_frame_1 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_names[n]), cv2.COLOR_RGB2BGR)
        lab_img = cv2.cvtColor(rgb_frame_1, cv2.COLOR_BGR2LAB)
        hsv_img = cv2.cvtColor(rgb_frame_1, cv2.COLOR_BGR2HSV)
        # _, threshold_otsu = cv2.threshold(lab_img[:, :, 1],0,255,cv2.THRESH_OTSU)
        # print(threshold_otsu)
        # rgb_frame_1[:, :, [0,2]] = 0
        lab_channel_imp = lab_img[:, :, 1]
        cv2.imshow('otsu', hsv_img[:, :, 0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        depth_frame_1 = cv2.imread("depth.png", cv2.CV_16UC1)  ##
        keypts_2d_1 = np.array(bboxes_d[os.path.basename(rgb_names[n])])
        depth_mask = np.zeros_like(depth_frame_1)
        for key_pt in keypts_2d_1:
            c, r = key_pt
            depth_mask[r - bbox_size:r + bbox_size, c - bbox_size:c + bbox_size] = 1

        # depth_frame_1 = np.multiply(depth_frame_1, depth_mask)
        rgbd_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_frame_1), 
            o3d.geometry.Image(depth_frame_1), 
            depth_scale=1, depth_trunc=10000, convert_rgb_to_intensity=False)

        target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_1, rgb_camera_intrinsic)

        # n + 1 th frame
        rgb_frame_2 = cv2.imread(args.data_root_path + rgb_names[n + skip_frames])
        depth_frame_2 = cv2.imread(args.data_root_path + depth_names[n + skip_frames], cv2.CV_16UC1)  ##
        keypts_2d_2 = np.array(bboxes_d[os.path.basename(rgb_names[n + skip_frames])])
        depth_mask = np.zeros_like(depth_frame_2)
        for key_pt in keypts_2d_2:
            c, r = key_pt
            depth_mask[r - bbox_size:r + bbox_size, c - bbox_size:c + bbox_size] = 1

        # depth_frame_2 = np.multiply(depth_frame_2, depth_mask)
        rgbd_2 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame_2),
         o3d.geometry.Image(depth_frame_2),
          depth_scale=1, depth_trunc=10000, convert_rgb_to_intensity=False)
        source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_2, rgb_camera_intrinsic)
        res = o3d.pipelines.registration.registration_icp(source, target, 0.002)

        T = T @ res.transformation
        merged_pcd += source.transform(T)
        # draw_registration_result(source, target, res.transformation)
        # o3d.visualization.draw_geometries([target])

        new_pose = np.r_[T[:-1, -1], tf.Rotation.from_matrix(T[:-1, :-1]).as_quat()]
        SE3_pose = np.vstack((SE3_pose, new_pose))
    
        if args.visualize or args.save:
            pass