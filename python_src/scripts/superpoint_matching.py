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
from matplotlib import pyplot as plt

import os
import sys
sys.path.append(os.pardir)

from utils.ransac_homography import *
from utils.icp import *


def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.
    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches[:2].T

def process_input_data(bboxes_path: str, associations_path: str) -> tuple[dict, list, list]:
    ''' Loads the input data for further use

    Arguments
    ---------
    - bboxes_path: Path to the pickle file containing the bounding box data for each rgb frame
    - associations_path: Path to the txt file conatining the associations for RGB and Depth frames from the dataset
    
    Returns
    -------
    - bboxes_data: {rgb_frame_path: bboxes} a dictionary containing the bounding boxes in each rgb frame as key-value pairs
    - rgb_frame_names: list containing filenames for all rgb frames from the associations file
    - depth_frame_names: list containing filenames for all depth frames from the associations file
    '''
    with open(bboxes_path, "rb") as f:
        bboxes_data = pickle.load(f)
    
    with open(associations_path, 'r') as f:
        depth_frame_names = []
        rgb_frame_names = []

        for line in f.readlines():
            _, rgb_path, _, depth_path = line.rstrip("\n").split(' ')
            if(os.path.basename(rgb_path) in list(bboxes_data.keys())):
                depth_frame_names.append(depth_path)
                rgb_frame_names.append(rgb_path)

    return bboxes_data, rgb_frame_names, depth_frame_names

def process_superpoint_feature_descriptors(superpoint_path: str) -> tuple[dict, dict]:

    with open(superpoint_path + 'points.pickle', "rb") as f:
        points_all = pickle.load(f)
    with open(superpoint_path + 'descriptors.pickle', "rb") as f:
        descriptors_all = pickle.load(f)

    
    return points_all, descriptors_all

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script matches descriptors from the superpoint algorithm''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/apples_big_2021-10-14-14-51-08_0/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
    args = parser.parse_args()

    points_all, descriptors_all = process_superpoint_feature_descriptors(args.data_root_path + 'superpoint/')

    with open(args.data_root_path + 'associations_rgbd.txt', 'r') as f:
        rgb_frame_names = []
        for line in f.readlines():
            _, rgb_path, _, _ = line.rstrip("\n").split(' ')
            rgb_frame_names.append(rgb_path)
    
    num_of_frames = len(rgb_frame_names)

    skip_frames = 10
    for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):
        rgb_frame_1 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_frame_names[n]), cv2.COLOR_RGB2BGR)
        rgb_frame_2 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_frame_names[n + skip_frames]), cv2.COLOR_RGB2BGR)

        points_1 = points_all[rgb_frame_names[n].split('/')[1]][:, :2]
        points_2 = points_all[rgb_frame_names[n + skip_frames].split('/')[1]][:, :2]

        descriptors_1 = descriptors_all[rgb_frame_names[n].split('/')[1]]
        descriptors_2 = descriptors_all[rgb_frame_names[n + skip_frames].split('/')[1]]

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
                    '../../eval_data/front/apples_big_2021-10-14-14-51-08_0/superpoint/{}.png'.format(int(n/skip_frames)), output_frame)
    
    cv2.destroyAllWindows()