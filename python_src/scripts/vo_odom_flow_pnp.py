import os
import sys
sys.path.append(os.pardir)
import argparse
from tqdm import tqdm

from utils.icp import *
from utils.utils import *
from superpoint_matching import nn_match_two_way
from utils.dataloader import DatasetOdometry, DatasetOdometryAll

from tqdm import tqdm

import cv2
import open3d as o3d
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.spatial.transform as tf


def getMatches(keypts_1: dict, keypts_2: dict, threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Get Matching SuperPoint Features
    
    Arguments
    ---------
    - keypts_1: Keypoints in image frame 1
    - keypts_2: Keypoints in image frame 2
    - threshold: Two way matching threshold

    Returns
    -------
    - points2d_1: Matched 2D keypoints in frame 1
    - points2d_2: Matched 2D keypoints in frame 2
    """
    points2d_1 = keypts_1['points'][:,:2]
    points2d_2 = keypts_2['points'][:,:2]

    descriptor_1 = keypts_1['desc']
    descriptor_2 = keypts_2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T, threshold).astype(np.int64)

    return points2d_1[M[:, 0]], points2d_2[M[:, 1]]


def convertTo3d(depth_frame: np.ndarray, keypoints_2d: np.ndarray, K: np.ndarray, depth_scale: float = 1000.0) -> np.ndarray:
    """Compute corresponding 3D keypoints for given 2D keypoints
    
    Arguments
    ---------
    - depth_frame: Depth Image for the corresponding frame
    - keypoints_2d: 2D keypoints detected from the corresponding RBG frame
    - K: Camera calibration matrix
    - depth_scale: conversion factor from depth pixel intensity to depth

    Returns
    -------
    - keypoint_3d: 3D keypoints
    """
    permute_col_2d = np.array([[0, 1],[1, 0]])
    permuted_keypoints = keypoints_2d @ permute_col_2d
    keypts_depth = depth_frame[permuted_keypoints[:, 0], permuted_keypoints[:, 1]] / depth_scale

    transformed_points = toHomogenous(keypoints_2d)
    ray_dir = transformed_points @ la.inv(K).T
    keypoints_3d = np.multiply(ray_dir, keypts_depth.reshape(-1, 1))
    
    return keypoints_3d

def argparser():
    """Argument Parser
    """
    parser = argparse.ArgumentParser(description='''PnP Pose Estimation using Superpoint Feature Tracking by LK Flow''')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize output')
    parser.add_argument('-d', '--debug', default=False, type=bool, help='Debug Flag')
    parser.add_argument('-p', '--plot', default=True, type=bool, help='Plot the odometry results')

    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = argparser()
    dataset_name = 'apples_big_2021-10-14-all/'

    if not "all" in dataset_name:
        dataset = DatasetOdometry(args.data_root_path + dataset_name)
    else:
        dataset = DatasetOdometryAll(args.data_root_path + dataset_name)
    
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        
    if "right" in args.data_root_path:
        max_depth = 5
        cam_intrinsics.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        max_depth = 10
        cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)

    K = cam_intrinsics.intrinsic_matrix.astype(np.double)
   
    T_rot =  (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ 
                np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])).T

    feature_params = {  'maxCorners' : 100,
                        'qualityLevel' : 0.3,
                        'minDistance' : 7,
                        'blockSize' : 7}
    lk_params = {'winSize': (15, 15),
                    'maxLevel': 2,
                    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    color = np.random.randint(0, 255, (100, 3))
            
    T = np.eye(4)
    poses = [odom_from_SE3(dataset[0]['timestamp'], T)]

    skip = args.skip_frames
    for n in tqdm(range(0, len(dataset) - skip, skip)):

        keypts_2d_1, keypts_2d_2 = getMatches(dataset[n], dataset[n + skip])   

        img_rgb_1 = dataset[n]['rgb']
        img_rgb_2 = dataset[n + skip]['rgb']

        img_gray_1 = cv2.cvtColor(img_rgb_1, cv2.COLOR_RGB2GRAY)
        img_gray_2 = cv2.cvtColor(img_rgb_2, cv2.COLOR_RGB2GRAY)

        p1 = keypts_2d_1.reshape(-1, 1, 2).astype(np.float32)

        p2, st, err = cv2.calcOpticalFlowPyrLK(img_gray_1, img_gray_2, p1, None, **lk_params)

        if p2 is not None:
            good_1 = p1[st == 1]
            good_2 = p2[st == 1]

        keypts_3d_1 = convertTo3d(dataset[n]['depth'], good_1.astype(np.int64), K)

        idx = np.where(
                        (keypts_3d_1[:, -1] <= max_depth) &
                        (keypts_3d_1[:, -1] > 0)
                        )[0]

        _, rvec, tvec, inliers = cv2.solvePnPRansac(keypts_3d_1[idx], good_2[idx].reshape(-1, 2), K, None) 

        T_local = la.inv(se3_to_SE3(rvec, tvec))
        
        T =  T @ T_local
        poses.append(odom_from_SE3(dataset[n + skip]['timestamp'], T, T_rot))

        if args.visualize:
            mask = np.zeros_like(img_rgb_1)
            for i, (new, old) in enumerate(zip(good_2[100:], good_1[100:])):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                img_rgb_2 = cv2.circle(img_rgb_2, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv2.add(img_rgb_2, mask)
            cv2.imshow('flow', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    poses = np.asarray(poses)

    cam_dir = 'front' if 'front' in args.data_root_path else 'right'

    np.savetxt(f"../../eval_data/{cam_dir}/{dataset_name}superpoint/LKFlow/poses_skip_{skip}.txt", poses)

    if args.plot:
        x = poses[:, 1]
        y = poses[:, 2]
        z = poses[:, 3]

        gt = np.loadtxt(args.data_root_path + "groundtruth.txt")
        gt_x = gt[:, 1]
        gt_y = gt[:, 2]
        gt_z = gt[:, 3]

        fig,(ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(y, z, 'r', label='superpoint-lkflow')
        ax1.plot(gt_y, gt_z ,'b', label='liosam')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('Z')
        ax1.legend()
        ax2.plot(x, z, 'r', label='superpoint-lkflow')
        ax2.plot(gt_x, gt_z, 'b', label='liosam')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.legend()
        ax3.plot(x, y, 'r', label='superpoint-lkflow')
        ax3.plot(gt_x, gt_y, 'b', label='liosam')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"../../eval_data/{cam_dir}/{dataset_name}superpoint/LKFlow/poses_skip_{skip}.png")
        plt.show()