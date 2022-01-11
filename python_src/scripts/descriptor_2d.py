#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import pickle
import argparse
from tqdm import tqdm

import cv2
import open3d as o3d
import scipy.spatial

import numpy as np
import numpy.linalg as la

import os
import sys
sys.path.append(os.pardir)

from utils.ransac_homography import *
from utils.utils import wrapTo2Pi

def process_input_data(bboxes_path: str, associations_path: str) -> tuple[dict, list]:
    ''' Loads the input data for further use

    Arguments
    ---------
    - bboxes_path: Path to the pickle file containing the bounding box data for each rgb frame
    - associations_path: Path to the txt file conatining the associations for RGB and Depth frames from the dataset
    
    Returns
    -------
    - bboxes_data: {rgb_frame_path: bboxes} a dictionary containing the bounding boxes in each rgb frame as key-value pairs
    - rgb_frame_names: list containing filenames for all rgb frames from the associations file
    '''
    with open(bboxes_path, "rb") as f:
        bboxes_data = pickle.load(f)
    
    with open(associations_path, 'r') as f:
        rgb_frame_names = []

        for line in f.readlines():
            _, rgb_path, _, _ = line.rstrip("\n").split(' ')
            if(os.path.basename(rgb_path) in list(bboxes_data.keys())):
                rgb_frame_names.append(rgb_path)

    return bboxes_data, rgb_frame_names

def find_k_nearest(keypts_2d: np.ndarray, k: int) -> (np.ndarray):
    ''' Compute the k nearest keypoints to the current keypoint of interest
    '''
    if keypts_2d.shape[0] > (k + 1):
        dist = scipy.spatial.distance.cdist(keypts_2d, keypts_2d)
        sort_ids = np.argsort(dist)
        return sort_ids[:, 1:k+1]
    else:
        raise RuntimeError("Not enough features to compute descriptor")

def compute_descriptor(keypoint: np.ndarray, knn_ids: np.ndarray):
    ''' Compute Descriptor for each 2d feature in the RGB frame

    Arguments
    ---------
    - keypoint: numpy array of all 2D keypoints in the RGB frame
    - knn_ids: indices corresponding to the k nearest neigbors of each 3D keypoint

    Returns
    -------
    - descriptor: [2D location, distance ratios, angles in 2D]
    - sort_idx: indices sorted according to increasing angle
    '''
    dist = la.norm(keypoint - knn_ids, 2, 1)
    
    dist_ratios = dist[:-1] / dist[-1]
    angles = np.arctan2(knn_ids[:-1, 1] - keypoint[1], knn_ids[:-1, 0] - keypoint[0]
        ) - np.arctan2(
            knn_ids[-1, 1] - keypoint[1], knn_ids[-1, 0] - keypoint[0]) 
    angles = np.array([wrapTo2Pi(-angle) for angle in angles]) / (2 * np.pi)

    sort_idx = np.argsort(angles)

    desc = np.r_[keypoint, dist_ratios[sort_idx], angles[sort_idx]]
    return desc, sort_idx

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script generates custom descriptors for 2D macro keypoints''')
    
    parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/bboxes.pickle', type=str,
        help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations.txt', type=str,
        help='Path to the associations file for RGB and Depth frames')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('--k', default=4, type=int, help='Number of nearest neighbors to use for descriptor')
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save_descriptor', default=False, type=bool, help='Save computed Descriptors as .npz file')
    args = parser.parse_args()

    bboxes_d, rgb_names = process_input_data(args.bboxes_path, args.associations_path)

    num_of_frames = len(rgb_names)
    skip_frames = 1
    for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):
        rgb_frame_1 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_names[n]), cv2.COLOR_RGB2BGR)
        keypts_2d_1 = np.array(bboxes_d[os.path.basename(rgb_names[n])])

        rgb_frame_2 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_names[n + skip_frames]), cv2.COLOR_RGB2BGR)
        keypts_2d_2 = np.array(bboxes_d[os.path.basename(rgb_names[n + skip_frames])])

        knn_ids_1 = find_k_nearest(keypts_2d_1, args.k)
        knn_ids_2 = find_k_nearest(keypts_2d_2, args.k)

        desc_1 = np.zeros((keypts_2d_1.shape[0], args.k * 2))
        desc_2 = np.zeros((keypts_2d_2.shape[0], args.k * 2))

        for i in range(keypts_2d_1.shape[0]):  
            desc_1[i], sort_idx = compute_descriptor(keypts_2d_1[i], keypts_2d_1[knn_ids_1[i]])
            # if args.visualize:
            #     image = rgb_names[n]
            #     delay = 1000
            #     cv2.namedWindow(image, cv2.WINDOW_NORMAL)
            #     cv2.circle(rgb_frame_1, (int(keypts_2d_1[i, 0]), int(keypts_2d_1[i, 1])), 10, (0, 0, 255), 3)
            #     cv2.imshow(image, rgb_frame_1)
            #     cv2.waitKey(delay)
            #     for j in range(args.k):
            #         cv2.circle(rgb_frame_1, (int(keypts_2d_1[knn_ids_1[i, j], 0]), int(keypts_2d_1[knn_ids_1[i, j], 1])), 10, (255, 0, 0), 3)
            #         cv2.imshow(image, rgb_frame_1)
            #         cv2.waitKey(delay)
            #     cv2.line(rgb_frame_1, (int(keypts_2d_1[i, 0]), int(keypts_2d_1[i, 1])), 
            #         (int(keypts_2d_1[knn_ids_1[i, -1], 0]), int(keypts_2d_1[knn_ids_1[i, -1], 1])), (0, 255, 0), 2)
            #     cv2.imshow(image, rgb_frame_1)
            #     cv2.waitKey(delay)
            #     for id in range(args.k - 1):
            #         cv2.line(rgb_frame_1, (int(keypts_2d_1[i, 0]), int(keypts_2d_1[i, 1])), 
            #             (int(keypts_2d_1[knn_ids_1[i, sort_idx[id]], 0]), int(keypts_2d_1[knn_ids_1[i, sort_idx[id]], 1])),
            #             (255, 0, 255), 2)
            #         cv2.imshow(image, rgb_frame_1)
            #         cv2.waitKey(delay)
            # cv2.destroyAllWindows()

        for i in range(keypts_2d_2.shape[0]):  
            desc_2[i], _ = compute_descriptor(keypts_2d_2[i], keypts_2d_2[knn_ids_2[i]])

        # cv2.namedWindow('matches', cv2.WINDOW_NORMAL)

        M = compute_matches(desc_1[:, 2:], desc_2[:, 2:], rgb_frame_1.shape[0], rgb_frame_1.shape[1])
        P1 = desc_1[:, :2].astype(int)
        P2 = desc_2[:, :2].astype(int)

        if(M.shape[0] > 0):
            H, M = compute_homography_ransac(P1, P2, M)

        w = rgb_frame_1.shape[1]
        img_match = np.concatenate((rgb_frame_1, rgb_frame_2), 1)

        for kp in desc_1:
            cv2.circle(img_match, (int(kp[0]), int(kp[1])), 10, (255, 0, 0), 2)

        for kp in desc_2:
            cv2.circle(img_match, (int(kp[0]) + w, int(kp[1])), 10, (0, 0, 255), 2)

        for ids in M:
            left_x = int(desc_1[ids[0], 0])
            left_y = int(desc_1[ids[0], 1])
            right_x = int(desc_2[ids[1], 0]) + w
            right_y = int(desc_2[ids[1], 1])

            cv2.line(img_match, (left_x, left_y),
                     (right_x, right_y), (0, 255, 255), 2)
        cv2.imshow('matches', img_match)
        # cv2.imwrite(
        #     '../../eval_data/custom_2d_desc/{}.png'.format(n), img_match)
        cv2.waitKey(0)