#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import cv2
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

import sys
sys.path.append('..')

from utils.ransac_homography import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    This script takes the keypoints kdtree object and generates custom descriptors for each keypoint
    ''')
    parser.add_argument('--bboxes_path', default='../../datasets/MineApple/bboxes.pickle', type=str,
                        help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('--descriptor_file', type=str,
                        help='Path to the numpy objects containing the descriptors in the dataset')
    parser.add_argument('-v', '--visualize', default=False,
                        type=bool, help='Visualize results')
    args = parser.parse_args()

    with open(args.bboxes_path, "rb") as f:
        bboxes_data = pickle.load(f)

    descriptors = np.load(args.descriptor_file, allow_pickle=True).item()

    images = list(descriptors.keys())

    for i in tqdm(range(len(images) - 1)):
        cv2.namedWindow('matches', cv2.WINDOW_NORMAL)

        M = compute_matches(
            descriptors[images[i]][:, 2:], descriptors[images[i + 1]][:, 2:])
        print(M.shape)
        P1 = descriptors[images[i]][:, :2].astype(int)
        P2 = descriptors[images[i+1]][:, :2].astype(int)

        print(P1.shape, P2.shape)

        if(M.shape[0] > 0):
            H, M = compute_homography_ransac(P1, P2, M)

        img_1 = cv2.imread(
            '../../datasets/MineApple/detection/train/images/' + images[i])
        img_2 = cv2.imread(
            '../../datasets/MineApple/detection/train/images/' + images[i + 1])

        w = img_1.shape[1]
        img_match = np.concatenate((img_1, img_2), 1)

        for kp in descriptors[images[i]]:
            cv2.circle(img_match, (int(kp[0]), int(kp[1])), 10, (255, 0, 0), 2)

        for kp in descriptors[images[i+1]]:
            cv2.circle(img_match, (int(kp[0]) + w,
                       int(kp[1])), 10, (0, 0, 255), 2)

        for ids in M:
            left_x = int(descriptors[images[i]][ids[0], 0])
            left_y = int(descriptors[images[i]][ids[0], 1])
            right_x = int(descriptors[images[i+1]][ids[1], 0]) + w
            right_y = int(descriptors[images[i+1]][ids[1], 1])

            cv2.line(img_match, (left_x, left_y),
                     (right_x, right_y), (0, 255, 255), 2)
        cv2.imshow('matches', img_match)
        cv2.imwrite(
            '../../eval_data/custom_2d_desc/{}.png'.format(i), img_match)
        cv2.waitKey(0)
