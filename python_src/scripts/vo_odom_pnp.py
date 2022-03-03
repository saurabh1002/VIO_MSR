import os
import sys; sys.path.append(os.pardir)
import argparse
from tqdm import tqdm

from utils.icp import *
from utils.utils import *
from superpoint_matching import nn_match_two_way
from utils.dataloader import DatasetOdometry, DatasetOdometryAll

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''VO using PnP algorithm with Superpoint Features''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=True, type=bool, help='Visualize output')
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

    T = np.eye(4)
    poses = []
    poses.append(odom_from_SE3(dataset[0]['timestamp'], T))

    min_dist = 0.01
    skip = args.skip_frames
    i = 0
    j = i + skip

    while True:
        try:
            print(f"Processed frames {i}/{len(dataset)}")
            keypts_2d_1, keypts_2d_2 = getMatches(dataset[i], dataset[j])    
            
            keypts_3d_1 = convertTo3d(dataset[i]['depth'], keypts_2d_1.astype(np.int64), K)
            keypts_3d_2 = convertTo3d(dataset[j]['depth'], keypts_2d_2.astype(np.int64), K)
            
            idx = np.where(
                    (keypts_3d_1[:, -1] <= max_depth) &
                    (keypts_3d_1[:, -1] > 0) &
                    (keypts_3d_2[:, -1] <= max_depth) &
                    (keypts_3d_2[:, -1] > 0)
                    )[0]

            _, rvec, tvec, inliers = cv2.solvePnPRansac(keypts_3d_1[idx], keypts_2d_2[idx], K, None)

            if la.norm(tvec) < min_dist:
                j = j + 1
                continue

            T_local = np.eye(4)
            R = cv2.Rodrigues(rvec)[0]
            T_local[:-1,:-1] = R
            T_local[:-1,-1] = tvec.reshape((3,))
            
            T =  T @ la.inv(T_local)
            poses.append(odom_from_SE3(dataset[j]['timestamp'], T))
            i = j
            j = j + skip
            if args.debug:
                print(f"Rotation: {R}\n")
                print(f"translation: {tvec}\n")
            
            if args.visualize:
                w = 640
                rgb_match_frame = np.concatenate((dataset[i]['rgb'], dataset[i + 1]['rgb']), 1)
                for kp in keypts_2d_1[idx]:
                    cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

                for kp in keypts_2d_2[idx]:
                    cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                            int(kp[1])), 3, (0, 0, 255), -1)
                output_frame = np.copy(rgb_match_frame)
            
                for kp_l, kp_r in zip(keypts_2d_1.astype(np.int64), keypts_2d_2.astype(np.int64)):
                    cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
                output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
                
                cv2.imshow("output", output_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except IndexError:
            break

    poses = np.asarray(poses)
    np.savetxt(f"../../eval_data/front/{dataset_name}superpoint/poses_pnp_skip{skip}.txt", poses)

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
        plt.savefig(f"../../eval_data/front/{dataset_name}superpoint/poses_pnp_skip{skip}.png")
        plt.show()
