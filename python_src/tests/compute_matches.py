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

def compute_matches(D1: np.ndarray, D2: np.ndarray) -> (np.ndarray):
    """
    Computes matches for two images using the descriptors, use the Lowe's criterea to determine the best match.
    Parameters
    ----------
    - D1 : descriptors for image 1 corners [num_corners x 128]
    - D2 : descriptors for image 2 corners [num_corners x 128]
 
    Returns
    ----------
    - M : [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints [num_matches x 2]
    """
    T = 0.5 # Threshold for maximum error in a match

    l_d1 = D1.shape[0]
    l_d2 = D2.shape[0]

    M = np.zeros((l_d1, 2), dtype=int)

    num_matches = 0
    # Compute Euclidean distance between each pair of descriptors in image 1 and image 2
    distances = cdist(D1, D2, 'euclidean')

    for i in range(distances.shape[0]):
        sort_idx = np.argsort(distances[i])
        # Find two keypoints in image 2 with least descriptor distance from ith keypoint in image 1
        idx_min_1 = sort_idx[0]
        idx_min_2 = sort_idx[1]
        min_1 = distances[i, idx_min_1]
        min_2 = distances[i, idx_min_2]
        # Lowe's criteria 1
        if min_1 < T:
            # Lowe's criteria 2
            if ((min_1 / min_2) < 0.5):
                M[num_matches] = np.array([int(i), int(idx_min_1)])
                num_matches += 1

    return M[:num_matches]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''
    This script takes the keypoints kdtree object and generates custom descriptors for each keypoint
    ''')
    parser.add_argument('--bboxes_path', default='../../datasets/MineApple/bboxes.pickle', type=str,
     help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('--descriptor_file', type=str,
     help='Path to the numpy objects containing the descriptors in the dataset')
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    args = parser.parse_args()

    with open(args.bboxes_path, "rb") as f:
        bboxes_data = pickle.load(f)

    descriptors = np.load(args.descriptor_file, allow_pickle=True).item()

    images = list(descriptors.keys())

    for i in tqdm(range(len(images) - 1)):
        # cv2.namedWindow('matches', cv2.WINDOW_NORMAL)

        M = compute_matches(descriptors[images[i]][:, 2:], descriptors[images[i + 1]][:, 2:])

        img_1 = cv2.imread('../../datasets/MineApple/detection/train/images/' + images[i])
        img_2 = cv2.imread('../../datasets/MineApple/detection/train/images/' + images[i + 1])

        w = img_1.shape[1]
        img_match = np.concatenate((img_1, img_2), 1)

        for kp in descriptors[images[i]]:
            cv2.circle(img_match, (int(kp[0]), int(kp[1])), 10, (255, 0, 0), 2)

        for kp in descriptors[images[i+1]]:
            cv2.circle(img_match, (int(kp[0]) + w, int(kp[1])), 10, (0, 0, 255), 2)

        for ids in M:
            left_x = int(descriptors[images[i]][ids[0], 0])
            left_y = int(descriptors[images[i]][ids[0], 1])
            right_x = int(descriptors[images[i+1]][ids[1], 0]) + w
            right_y = int(descriptors[images[i+1]][ids[1], 1])

            cv2.line(img_match, (left_x, left_y), (right_x, right_y), (0, 255, 255), 2)
        # cv2.imshow('matches', img_match)
        cv2.imwrite('../../eval_data/custom_2d_desc/{}.png'.format(i), img_match)
        # cv2.waitKey(100)