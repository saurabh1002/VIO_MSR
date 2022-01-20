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

from utils.ransac_homography import *
from utils.icp import *

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

def get_depth_mask(depth_frame_shape: tuple, keypoint: np.ndarray) -> (np.ndarray):
    '''This function creates a depth mask for a keypoint pixel

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

def get_keypoints(depth_frame: np.ndarray, keypts_2d: np.ndarray, rgb_camera_intrinsic: o3d._pybind_cuda,
    depth_scale: float = 1000) -> tuple[np.ndarray, np.ndarray]:
    ''' Computes 3D keypoints from the detection centers in the RGB frame and associated depth frame
    
    Arguments
    ---------
    - depth_frame: Depth frame to be processed
    - keypts_2d: 2D keypoints in the frame
    - rgb_camera_intrinsic: Intrinsic parameters of the RGB camera in open3d format
    - depth_scale: scale factor for the depth camera [m <-> pixel value]

    Returns
    -------
    - keypts_2d: 2D keypoints in the frame
    - keypts_3d: corresponding 3D keypoints in the frame

    '''
    keypts_3d = []
    pcl_gen_flag = []

    for keypt in keypts_2d:
        depth_mask = get_depth_mask(depth_frame.shape, keypt)
        depth_image = o3d.geometry.Image(np.multiply(depth_frame, depth_mask))
        pcl = o3d.geometry.PointCloud.create_from_depth_image(depth_image, rgb_camera_intrinsic, depth_scale=depth_scale, depth_trunc=3)

        # Check if 2d keypoint could be succesfully projected to 3d
        pcl_gen_flag.append(pcl.has_points())
        if(pcl.has_points()):
            keypts_3d.append(np.array(pcl.points)[0])
    
    keypts_2d = np.array(keypts_2d)[pcl_gen_flag]
    keypts_3d = np.array(keypts_3d)

    return keypts_2d, keypts_3d

def compute_hist_3d(keypts_3d: np.ndarray) -> (np.ndarray):
    '''Compute histogram based on number of detections in each octant around current keypoint

    Argument
    --------
    - keypts_3d: array of all 3d keypoints from the RGB-D frame of interest

    Returns
    -------
    - hist_3d: a histogram for each feature in the RGB-D frame of interest
    '''
    hist_3d = np.zeros((keypts_3d.shape[0], 8))
    for i, kp in enumerate(keypts_3d):
        # Relative coordinates of all keypoints wrt the current keypoint of interest
        rel_kps = np.delete(keypts_3d, i, 0) - kp
        x, y, z = rel_kps[:, 0], rel_kps[:, 1], rel_kps[:, 2]

        # Compute count of keypoints in each octant
        hist_3d[i, 0] = np.sum((x >= 0) & (y >= 0) & (z >= 0))
        hist_3d[i, 1] = np.sum((x >= 0) & (y >= 0) & (z < 0))
        hist_3d[i, 2] = np.sum((x >= 0) & (y < 0) & (z >= 0))
        hist_3d[i, 3] = np.sum((x >= 0) & (y < 0) & (z < 0))
        hist_3d[i, 4] = np.sum((x < 0) & (y >= 0) & (z >= 0))
        hist_3d[i, 5] = np.sum((x < 0) & (y >= 0) & (z < 0))
        hist_3d[i, 6] = np.sum((x < 0) & (y < 0) & (z >= 0))
        hist_3d[i, 7] = np.sum((x < 0) & (y < 0) & (z < 0))

    return hist_3d / keypts_3d.shape[0]

def compute_descriptor(keypts_2d: np.ndarray, keypoints_3d: np.ndarray, hist: np.ndarray) -> (np.ndarray):
    ''' Compute Descriptor for each 3d feature in the RGB-D frame
        descriptor = [2D location, 3D location, histogram]
    '''
    descriptors = np.zeros((keypoints_3d.shape[0], 5 + hist.shape[1]))
    for i, kp3d in enumerate(keypoints_3d):
        descriptors[i] = np.r_[keypts_2d[i], kp3d, hist[i]]
    return descriptors

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script generates custom descriptors for 3D macro keypoints''')

    # Dataset paths
    parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples/detection.pickle', type=str,
        help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples/associations_rgbd.txt', type=str,
        help='Path to the associations file for RGB and Depth frames')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
    args = parser.parse_args()

    bboxes_d, rgb_names, depth_names = process_input_data(args.bboxes_path, args.data_root_path + 'associations_rgbd.txt')
    num_of_frames = len(rgb_names)

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if args.data_root_path == "../../datasets/phenorob/images_apples_right/":
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif args.data_root_path == '../../datasets/phenorob/images_apples/':
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    
    # [x, y, z, qx, qy, qz, qw]
    SE3_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    T = np.eye(4)

    skip_frames = 1
    for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):
        # n th frame
        rgb_frame_1 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_names[n]), cv2.COLOR_RGB2BGR)
        depth_frame_1 = cv2.imread(args.data_root_path + depth_names[n], cv2.CV_16UC1)  ##
        keypts_2d_1 = np.array(bboxes_d[os.path.basename(rgb_names[n])])
        keypts_2d_1, keypts_3d_1 = get_keypoints(depth_frame_1, keypts_2d_1, rgb_camera_intrinsic)
        pcl_1 = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_frame_1), rgb_camera_intrinsic, depth_scale=1000, depth_trunc=3)

        # n + 1 th frame
        rgb_frame_2 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_names[n + skip_frames]), cv2.COLOR_RGB2BGR)
        depth_frame_2 = cv2.imread(args.data_root_path + depth_names[n + skip_frames], cv2.CV_16UC1)  ##
        keypts_2d_2 = np.array(bboxes_d[os.path.basename(rgb_names[n + skip_frames])])
        keypts_2d_2, keypts_3d_2 = get_keypoints(depth_frame_2, keypts_2d_2, rgb_camera_intrinsic)
        pcl_2 = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_frame_2), rgb_camera_intrinsic, depth_scale=1000, depth_trunc=3)
        
        hist_1 = compute_hist_3d(keypts_3d_1)
        hist_2 = compute_hist_3d(keypts_3d_2)

        descriptors_1 = compute_descriptor(keypts_2d_1, keypts_3d_1, hist_1)
        descriptors_2 = compute_descriptor(keypts_2d_2, keypts_3d_2, hist_2)

        # M = compute_matches(descriptors_1[:, 5:], descriptors_2[:, 5:], rgb_frame_1.shape[0], rgb_frame_1.shape[1])
        M = nn_match_two_way(descriptors_1[:, 5:].T, descriptors_2[:, 5:].T, 0.7)

        P1_2d = descriptors_1[:, :2].astype(int)
        P2_2d = descriptors_2[:, :2].astype(int)

        P1_3d = descriptors_1[:, 2:5]
        P2_3d = descriptors_2[:, 2:5]

        if(M.shape[0] > 0):
            H, M = compute_homography_ransac(P1_2d, P2_2d, M)

        T_new = icp_known_corresp(P1_3d.T, P2_3d.T, M[:, 0], M[:, 1])

        corr_ind = np.array([[np.where((np.array(pcl_2.points) == tuple(P2_3d[M[i, 1]])).all(axis=1))[0][0], 
            np.where((np.array(pcl_1.points) == tuple(P1_3d[M[i, 0]])).all(axis=1))[0][0]] for i in range(len(M))])
        res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pcl_2, pcl_1, o3d.utility.Vector2iVector(corr_ind), 0.05)

        res = o3d.pipelines.registration.registration_icp(pcl_2, pcl_1, 0.025, res.transformation)
        
        T_new = res.transformation

        # o3d.visualization.draw_geometries([pcl_1])

        T = T @ T_new

        new_pose = np.r_[T[:-1, -1], tf.Rotation.from_matrix(T[:-1, :-1]).as_quat()]
        SE3_pose = np.vstack((SE3_pose, new_pose))
    
        if args.visualize or args.save:
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
                cv2.namedWindow('matches')
                cv2.imshow('matches', rgb_match_frame)
                cv2.waitKey(250)
            if args.save:
                cv2.imwrite(
                    '../../eval_data/custom_3d_desc/{}.png'.format(int(n/skip_frames)), rgb_match_frame)
    
    cv2.destroyAllWindows()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_title('XY pose')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.axis('equal')
    ax1.plot(SE3_pose[:, 0], SE3_pose[:, 1])
    
    ax2.set_title('YZ pose')
    ax2.set_xlabel('Y [m]')
    ax2.set_ylabel('Z [m]')
    ax2.axis('equal')
    ax2.plot(SE3_pose[:, 1], SE3_pose[:, 2])
    
    ax3.set_title('XZ pose')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Z [m]')
    ax3.axis('equal')
    ax3.plot(SE3_pose[:, 0], SE3_pose[:, 2])
    plt.show()