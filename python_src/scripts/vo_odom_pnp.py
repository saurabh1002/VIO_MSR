import os

import sys; sys.path.append(os.pardir)
import argparse
from tqdm import tqdm

from utils.icp import *
from utils.utils import *
from superpoint_matching import nn_match_two_way
from utils.dataloader import DatasetOdometry, DatasetOdometryAll

from descriptor_3d_hist import *

import cv2
import open3d as o3d
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def getMatches(data_1: dict, data_2: dict, 
                K: np.ndarray,
                superpoint_threshold: float = 0.7, 
                fast_threshold: float = 5, 
                type: str = 'superpoint'
               ):
    if type == 'superpoint':
        descriptor_1 = data_1['desc']
        descriptor_2 = data_2['desc']

        M = nn_match_two_way(descriptor_1.T, descriptor_2.T, superpoint_threshold).astype(np.int64)

        keypts_2d_1 = data_1['points'][:,:2][M[:, 0]]
        keypts_2d_2 = data_2['points'][:,:2][M[:, 1]]

        keypts_3d_1 = convertTo3d(data_1['depth'], keypts_2d_1.astype(np.int64), K)
        keypts_3d_2 = convertTo3d(data_2['depth'], keypts_2d_2.astype(np.int64), K)
            
        return keypts_2d_1, keypts_2d_2, keypts_3d_1, keypts_3d_2, True

    elif type == 'ORB':
        img_1 = data_1['rgb']
        img_2 = data_2['rgb']
        orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, fastThreshold=fast_threshold)
        keypts_2d_1, descriptor_1 = orb.detectAndCompute(img_1, None)
        keypts_2d_2, descriptor_2 = orb.detectAndCompute(img_2, None)
        
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
        matches = matcher.match(descriptor_1, descriptor_2)

        points2d_1 = []
        points2d_2 = []
        for match in matches:
            points2d_1.append(keypts_2d_1[match.queryIdx].pt)
            points2d_2.append(keypts_2d_2[match.trainIdx].pt)
        
        points2d_1 = np.array(points2d_1)
        points2d_2 = np.array(points2d_2)

        keypts_3d_1 = convertTo3d(data_1['depth'], points2d_1.astype(np.int64), K)
        keypts_3d_2 = convertTo3d(data_2['depth'], points2d_2.astype(np.int64), K)
        

        return points2d_1, points2d_2, keypts_3d_1, keypts_3d_2, True

    elif type == '3Dhist':      
        if data_1['det'] == None or data_2['det'] == None:
            return None, None, None, None, False

        keypts_2d_1 = np.array(data_1['det'])[:, :2]
        keypts_2d_2 = np.array(data_2['det'])[:, :2]

        keypts_3d_1 = convertTo3d(data_1['depth'], keypts_2d_1.astype(np.int64), K)
        keypts_3d_2 = convertTo3d(data_2['depth'], keypts_2d_2.astype(np.int64), K)
        
        hist_1 = compute_hist_3d(keypts_3d_1)
        hist_2 = compute_hist_3d(keypts_3d_2)

        descriptors_1 = compute_descriptor(keypts_2d_1, keypts_3d_1, hist_1)
        descriptors_2 = compute_descriptor(keypts_2d_2, keypts_3d_2, hist_2)

        M = compute_matches(descriptors_1[:, 5:], descriptors_2[:, 5:], None, None)

        if(M.shape[0] > 0):
            H, M = compute_homography_ransac(keypts_2d_1, keypts_2d_2, M)

        return keypts_2d_1[M[:, 0]], keypts_2d_2[M[:, 1]], keypts_3d_1[M[:, 0]], keypts_3d_2[M[:, 1]], True

def convertTo3d(depth_frame: np.ndarray, keypoints_2d: np.ndarray, K: np.ndarray) -> (np.ndarray):
    depth_factor = 1000
    permute_col_2d = np.array([[0, 1], [1, 0]])
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
    parser.add_argument('-t', '--type', required=True, type=str, help='Type of the features and descriptor'),
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize output')
    parser.add_argument('-d', '--debug', default=False, type=bool, help='Debug Flag')
    parser.add_argument('-p', '--plot', default=True, type=bool, help='Plot the odometry results')

    args = parser.parse_args()
    skip = args.skip_frames
    method = args.type

    dataset_name = 'apples_big_2021-10-14-all/'
    if not "all" in dataset_name:
        dataset = DatasetOdometry(args.data_root_path + dataset_name, method)
    else:
        dataset = DatasetOdometryAll(args.data_root_path + dataset_name, method)
   
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    
    if "right" in args.data_root_path:
        max_depth = 5
        cam_intrinsics.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        max_depth = 10
        cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)

    K = cam_intrinsics.intrinsic_matrix.astype(np.double)

    T_rot =  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    T = np.eye(4)
    poses = []

    poses.append(odom_from_SE3(dataset.timestamps[0], T))

    min_dist_flag = False
    min_dist = 0.75

    i = 0
    j = i + skip

    while True:
        try:
            keypts_2d_1, keypts_2d_2, keypts_3d_1, keypts_3d_2, det_flag = \
                getMatches(dataset[i], dataset[j], K, type=method)

            if not det_flag or min(len(keypts_2d_1), len(keypts_2d_2)) < 6:
                i = i + skip
                j = j + skip
                continue
                # keypts_2d_1, keypts_2d_2, keypts_3d_1, keypts_3d_2, det_flag = \
                # getMatches(dataset[i], dataset[j], K, type='ORB')

            idx = np.where(
                        (keypts_3d_1[:, -1] <= max_depth) & (keypts_3d_1[:, -1] > 0) &
                        (keypts_3d_2[:, -1] <= max_depth) & (keypts_3d_2[:, -1] > 0)
                    )[0]
            
            _, rvec, tvec, inliers = cv2.solvePnPRansac(keypts_3d_1[idx].astype(np.float64), keypts_2d_2[idx].astype(np.float64), K, None)

            if min_dist_flag:
                if la.norm(tvec) < min_dist and la.norm(rvec) < np.pi / 18:
                    j = j + 1
                    continue
            
            T_local = la.inv(se3_to_SE3(rvec, tvec))
            
            T =  T @ T_local
            poses.append(odom_from_SE3(dataset[j]['timestamp'], T))

            i = j
            if not min_dist_flag:
                j = j + skip
            else:
                j = j + 1

            if args.debug:
                print(f"Rotation: {T[:3, :3]}\n")
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
                # output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
                
                cv2.imshow("output", rgb_match_frame)
                cv2.waitKey(1)

        except IndexError:
            print('Exiting')
            break

    cv2.destroyAllWindows()

    poses = np.asarray(poses)

    cam_dir = 'front' if 'front' in args.data_root_path else 'right'

    if args.plot:
        x = poses[:, 1]
        y = poses[:, 2]
        z = poses[:, 3]

        gt = np.loadtxt(args.data_root_path + "groundtruth.txt")
        gt_x = gt[:, 1]
        gt_y = gt[:, 2]
        gt_z = gt[:, 3]

        fig,(ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(y, z, 'r', label='Estimated')
        ax1.plot(gt_y, gt_z ,'b', label='LIDAR Poses')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('Z')
        ax1.legend()
        ax2.plot(x, z, 'r', label='Estimated')
        ax2.plot(gt_x, gt_z, 'b', label='LIDAR Poses')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.legend()
        ax3.plot(x, y, 'r', label='Estimated')
        ax3.plot(gt_x, gt_y, 'b', label='LIDAR Poses')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()

        plt.tight_layout()
        if not min_dist_flag:
            plt.savefig(f"../../eval_data/{cam_dir}/{dataset_name}{method}/PnP/poses_skip_{skip}.png")
        else:
            plt.savefig(f"../../eval_data/{cam_dir}/{dataset_name}{method}/PnP/poses_min_dist_{min_dist}_10deg.png")
        plt.show()

    if not min_dist_flag:
        np.savetxt(f"../../eval_data/{cam_dir}/{dataset_name}{method}/PnP/poses_skip_{skip}.txt", poses)
    else:
        np.savetxt(f"../../eval_data/{cam_dir}/{dataset_name}{method}/PnP/poses_min_dist_{min_dist}_10deg.txt", poses)