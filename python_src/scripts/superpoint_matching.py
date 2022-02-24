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
import sys; sys.path.append(os.pardir)

from utils.dataloader import DatasetOdometry, DatasetOdometryAll
from utils.ransac_homography import *
from utils.icp import *


def getMatches(keypts_1: dict, keypts_2: dict, threshold: float = 0.7):
    
    points2d_1 = keypts_1['points'][:,:2]
    points2d_2 = keypts_2['points'][:,:2]

    descriptor_1 = keypts_1['desc']
    descriptor_2 = keypts_2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T, threshold).astype(np.int64)

    return points2d_1[M[:, 0]], points2d_2[M[:, 1]]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script matches descriptors from the superpoint algorithm''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/', type=str,
        help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
    args = parser.parse_args()

    dataset_name = 'apples_big_2021-10-14-all/'
    if not "all" in dataset_name:
        dataset = DatasetOdometry(args.data_root_path + dataset_name)
    else:
        dataset = DatasetOdometryAll(args.data_root_path + dataset_name)

    skip = args.skip_frames
    progress_bar = tqdm(range(0, len(dataset) - skip, skip))
    for i in progress_bar:
        keypts_2d_1, keypts_2d_2 = getMatches(dataset[i], dataset[i + skip])

        if args.visualize or args.save:            
            w = 640
            rgb_match_frame = np.concatenate((dataset[i]['rgb'], dataset[i + skip]['rgb']), 1)
            for kp in keypts_2d_1:
                cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

            for kp in keypts_2d_2:
                cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                        int(kp[1])), 3, (0, 0, 255), -1)
            output_frame = np.copy(rgb_match_frame)
        
            for kp_l, kp_r in zip(keypts_2d_1[:50].astype(int), keypts_2d_2[:50].astype(int)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            for kp_l, kp_r in zip(keypts_2d_1.astype(int), keypts_2d_2.astype(int)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            if args.visualize:
                cv2.imshow("output", output_frame)
                cv2.waitKey(0)
            if args.save:
                cv2.imwrite(
                    f'../../eval_data/right/{dataset_name}superpoint/{int(i / skip)}.png', output_frame)
    
    cv2.destroyAllWindows()