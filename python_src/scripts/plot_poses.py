import os

import sys; sys.path.append(os.pardir)
import argparse
from tqdm import tqdm

from utils.icp import *
from utils.utils import *

from descriptor_3d_hist import *

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''VO using PnP algorithm with Superpoint Features''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-t', '--type', required=True, type=str, help='Type of the features and descriptor'),
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")

    args = parser.parse_args()
    skip = args.skip_frames
    method = args.type

    cam_dir = 'front' if 'front' in args.data_root_path else 'right'
    
    dataset_name = 'apples_big_2021-10-14-all/'
    T_rot =  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    poses_combined = np.loadtxt(f"../../eval_data/{cam_dir}/{dataset_name}{method}/PnP/poses_combined_min_dist_1_10deg.txt")

    poses_combined = poses_combined[:, 1:4] @ T_rot
    com_x = poses_combined[:, 0]
    com_y = poses_combined[:, 1]
    com_z = poses_combined[:, 2]

    # poses_vo = np.loadtxt(f"../../eval_data/{cam_dir}/{dataset_name}{method}/PnP/poses_min_dist_1_10deg.txt")
    poses_vo = np.loadtxt(f'/home/dhagash/Downloads/eval_data/front/apples_big_2021-10-14-all/superpoint/PnP/poses_min_dist_1_10deg.txt')
    
    poses_vo = poses_vo[:, 1:4] @ T_rot
    vo_x = poses_vo[:, 0]
    vo_y = poses_vo[:, 1]
    vo_z = poses_vo[:, 2]

    gt = np.loadtxt(args.data_root_path + "groundtruth.txt")
    gt_x = gt[:, 1]
    gt_y = gt[:, 2]
    gt_z = gt[:, 3]

    fig,(ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(com_y, com_z, 'g', label='Combined EKF Estimated')
    ax1.plot(vo_y, vo_z, 'r', label='VO')
    ax1.plot(gt_y, gt_z ,'b', label='LIDAR Poses')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    ax1.legend()

    ax2.plot(com_x, com_z, 'g', label='Combined EKF Estimated')
    ax2.plot(vo_x, vo_z, 'r', label='VO')
    ax2.plot(gt_x, gt_z, 'b', label='LIDAR Poses')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.legend()

    ax3.plot(com_x, com_y, 'g', label='Combined EKF Estimated')
    ax3.plot(vo_x, vo_y, 'r', label='VO')
    ax3.plot(gt_x, gt_y, 'b', label='LIDAR Poses')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()


    plt.tight_layout()
    plt.savefig(f"../../eval_data/{cam_dir}/{dataset_name}{method}/PnP/poses_combined_min_dist_{skip}_10deg.png")
    plt.show()