#!/usr/bin/env python

# ==============================================================================

# @Authors: Dhagash Desai, Saurabh Gupta
# @email: s7dhdesa@uni-bonn.de, s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import pickle
import argparse
from tqdm import tqdm

import cv2
import numpy as np

import os
import sys
sys.path.append(os.pardir)

from utils.ransac_homography import *
from utils.icp import *


def process_superpoint_feature_descriptors(superpoint_path: str) -> tuple[dict, dict]:

    with open(superpoint_path + 'points.pickle', "rb") as f:
        points_all = pickle.load(f)
    with open(superpoint_path + 'descriptors.pickle', "rb") as f:
        descriptors_all = pickle.load(f)
    
    return points_all, descriptors_all

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script matches descriptors from the superpoint algorithm''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/', type=str,
        help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
    args = parser.parse_args()

    dataset_name = 'apples_big_2021-10-14-14-51-08_0/'

    points_all, descriptors_all = process_superpoint_feature_descriptors(args.data_root_path + dataset_name + 'superpoint/')

    with open(args.data_root_path + 'associations_rgbd.txt', 'r') as f:
        rgb_frame_names = []
        for line in f.readlines():
            _, rgb_path, _, _ = line.rstrip("\n").split(' ')
            rgb_frame_names.append(rgb_path)
    
    num_of_frames = len(rgb_frame_names)

    skip = args.skip_frames
    for n in tqdm(range(0, num_of_frames - skip, skip)):
        rgb_frame_1 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_frame_names[n]), cv2.COLOR_RGB2BGR)
        rgb_frame_2 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_frame_names[n + skip]), cv2.COLOR_RGB2BGR)

        points_1 = points_all[rgb_frame_names[n].split('/')[1]][:, :2]
        points_2 = points_all[rgb_frame_names[n + skip].split('/')[1]][:, :2]

        descriptors_1 = descriptors_all[rgb_frame_names[n].split('/')[1]]
        descriptors_2 = descriptors_all[rgb_frame_names[n + skip].split('/')[1]]

        M = nn_match_two_way(descriptors_1.T, descriptors_2.T, 0.4)
        M = M.astype(np.int32)

        # if M.shape[0] > 0:
        #     H, M = compute_homography_ransac(points_1, points_2, M)

        if args.visualize or args.save:            
            w = rgb_frame_1.shape[1]
            rgb_match_frame = np.concatenate((rgb_frame_1, rgb_frame_2), 1)
            for kp in points_1:
                cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

            for kp in points_2:
                cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                        int(kp[1])), 3, (0, 0, 255), -1)
            output_frame = np.copy(rgb_match_frame)
        
            for kp_l, kp_r in zip(points_1[M[:50, 0]].astype(int), points_2[M[:50, 1]].astype(int)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            for kp_l, kp_r in zip(points_1[M[50:, 0]].astype(int), points_2[M[50:, 1]].astype(int)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            if args.visualize:
                cv2.imshow("output", output_frame)

            if args.save:
                cv2.imwrite(
                    f'../../eval_data/front/{dataset_name}superpoint/{int(n / skip)}.png', output_frame)
    
    cv2.destroyAllWindows()