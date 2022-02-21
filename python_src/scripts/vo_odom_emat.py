import os
import argparse
from tqdm import tqdm
import sys; sys.path.append(os.pardir)

from utils.utils import *
from utils.dataloader import DatasetOdometry, DatasetOdometryAll
from superpoint_matching import nn_match_two_way

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

def get_relative_orientation(points_1: np.ndarray, points_2: np.ndarray, K: np.ndarray, K_inv: np.ndarray):
    prob = 0.999
    threshold = 1
    method = cv2.RANSAC 

    e, mask = cv2.findEssentialMat(points_1,
                                   points_2,
                                   K_inv,
                                   method=method,
                                   prob=prob,
                                   threshold=threshold)
    # U,S,V = np.linalg.svd(e)

    _,R,t,_= cv2.recoverPose(e, points_1[mask], points_2[mask], K)

    # print(mask1.shape)
    # pts = (R @ np.hstack((points_1, np.ones((points_1.shape[0], 1)))).T + t).T
    # err = la.norm(pts[:,:-1] - points_2) / (points_2.shape[0])

    return R, t.reshape((3,))

def getMatches(keypts_1: dict, keypts_2: dict, threshold: float = 0.7):
    
    points2d_1 = keypts_1['points'][:,:2]
    points2d_2 = keypts_2['points'][:,:2]

    descriptor_1 = keypts_1['desc']
    descriptor_2 = keypts_2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T, threshold).astype(np.int64)

    return points2d_1[M[:, 0]], points2d_2[M[:, 1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Odometry using superpoint matching''')

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
        cam_intrinsics.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)

    K = cam_intrinsics.intrinsic_matrix.astype(np.double)
    K_inv = la.inv(K).astype(np.double)

    T = np.eye(4)
    poses = []

    skip = args.skip_frames
    progress_bar = tqdm(range(0, len(dataset) - skip, skip))
    for i in progress_bar:
        keypts_2d_1, keypts_2d_2 = getMatches(dataset[i], dataset[i + 1])

        # idx = np.where(la.norm(keypts_2d_1 - keypts_2d_2, axis=1) > 1.3) [0]
        if (np.linalg.norm(keypts_2d_1 - keypts_2d_2) > 70):
            R, tvec = get_relative_orientation(keypts_2d_1, keypts_2d_2, K, K_inv)
            
            T_local = np.eye(4)
            T_local[:-1, :-1] = R
            T_local[:-1, -1] = tvec
            T =  T @ T_local

            if args.debug:
                print(f"Rotation: {R}\n")
                print(f"translation: {tvec}\n")

            if args.visualize:
                w = 640
                rgb_match_frame = np.concatenate((dataset[i]['rgb'], dataset[i + 1]['rgb']), 1)
                for kp in keypts_2d_1:
                    cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

                for kp in keypts_2d_2:
                    cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                            int(kp[1])), 3, (0, 0, 255), -1)
                output_frame = np.copy(rgb_match_frame)
            
                for kp_l, kp_r in zip(keypts_2d_1.astype(np.int64), keypts_2d_2.astype(np.int64)):
                    cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
                output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
                
                cv2.imshow("output", output_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()      

    poses = np.asarray(poses)
    np.savetxt(f"../../eval_data/front/{dataset_name}poses_EMat_skip{skip}.txt", poses)

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
        ax1.legend()
        ax2.plot(z, -y, 'r', label='Estimated')
        ax2.plot(gt_x, gt_z, 'b', label='Ground Truth')
        ax2.legend()
        ax3.plot(z, -x, 'r', label='Estimated')
        ax3.plot(gt_x, gt_y, 'b', label='Ground Truth')
        ax3.legend()

        plt.show()