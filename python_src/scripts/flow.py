from ast import arg
import os
import sys
from unittest import skip; sys.path.append(os.pardir)
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


def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:-1, :-1]).as_quat()
    return list(np.r_[t, origin, rot_quat])


def getMatches(keypts_1: dict, keypts_2: dict, threshold: float = 0.7):
    
    points2d_1 = keypts_1['points'][:,:2]
    points2d_2 = keypts_2['points'][:,:2]

    descriptor_1 = keypts_1['desc']
    descriptor_2 = keypts_2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T, threshold).astype(np.int64)

    return points2d_1[M[:, 0]], points2d_2[M[:, 1]]


def convertTo3d(depth_frame: np.ndarray, keypoints_2d: np.ndarray, K: np.ndarray):

    depth_factor = 1000
    permute_col_2d = np.array([[0, 1],[1, 0]])
    permuted_keypoints = keypoints_2d @ permute_col_2d
    keypts_depth = depth_frame[permuted_keypoints[:, 0], permuted_keypoints[:, 1]] / depth_factor

    transformed_points = toHomogenous(keypoints_2d)
    ray_dir = transformed_points @ la.inv(K).T
    keypoints_3d = np.multiply(ray_dir, keypts_depth.reshape(-1, 1))
    
    return keypoints_3d

parser = argparse.ArgumentParser(description='''This script generates custom descriptors for 3D macro keypoints''')

# Dataset paths
parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                    type=str, help='Path to the root directory of the dataset')
parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize output')
parser.add_argument('-d', '--debug', default=False, type=bool, help='Debug Flag')
parser.add_argument('-p', '--plot', default=True, type=bool, help='Plot the odometry results')

args = parser.parse_args()

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

feature_params = {  'maxCorners' : 100,
                    'qualityLevel' : 0.3,
                    'minDistance' : 7,
                    'blockSize' : 7}

lk_params = {'winSize': (15, 15),
                  'maxLevel': 2,
                  'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

color = np.random.randint(0, 255, (100, 3))
        
T = np.eye(4)
poses = []
poses.append(odom_from_SE3(dataset[0]['timestamp'], T))

skip_frames = args.skip_frames
for n in tqdm(range(0, len(dataset) - skip_frames, skip_frames)):

    keypts_2d_1, keypts_2d_2 = getMatches(dataset[n], dataset[n + skip_frames])   

    img_rgb_1 = dataset[n]['rgb']

    img_rgb_2 = dataset[n + skip_frames]['rgb']

    img_gray_1 = cv2.cvtColor(img_rgb_1, cv2.COLOR_RGB2GRAY)
    img_gray_2 = cv2.cvtColor(img_rgb_2, cv2.COLOR_RGB2GRAY)

    # p1 = cv2.goodFeaturesToTrack(img_gray_1, mask = None, **feature_params)
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

    T_local = np.eye(4)
    R = cv2.Rodrigues(rvec)[0]
    T_local[:-1,:-1] = R
    T_local[:-1,-1] = tvec.reshape((3,))
    
    T =  T @ la.inv(T_local)
    poses.append(odom_from_SE3(dataset[n + skip_frames]['timestamp'], T))

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
    flow = cv2.calcOpticalFlowFarneback(img_gray_1, img_gray_2, None, 0.5, 3, 15, 3, 15, 1.2, 0)
    print(flow.shape)
    
poses = np.asarray(poses)
np.savetxt(f"../../eval_data/front/{dataset_name}superpoint/poses_flowpnp_skip{skip_frames}.txt", poses)

if args.plot:
    x = poses[:, 1]
    y = poses[:, 2]
    z = poses[:, 3]

    gt = np.loadtxt(args.data_root_path + "groundtruth.txt")
    gt_x = gt[:, 1]
    gt_y = gt[:, 2]
    gt_z = gt[:, 3]

    fig,(ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(-x, -y, 'r', label='Estimated')
    ax1.plot(gt_y, gt_z ,'b', label='Ground Truth')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    ax1.legend()
    ax2.plot(z, -y, 'r', label='Estimated')
    ax2.plot(gt_x, gt_z, 'b', label='Ground Truth')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.legend()
    ax3.plot(z, -x, 'r', label='Estimated')
    ax3.plot(gt_x, gt_y, 'b', label='Ground Truth')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f"../../eval_data/front/{dataset_name}superpoint/poses_flowpnp_skip{skip_frames}.png")
    plt.show()


    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hsv = np.zeros_like(img_rgb_1)
    # hsv[..., 1] = 255
    # hsv[..., 0] = ang * 90 / np.pi
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('dense flow', bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()