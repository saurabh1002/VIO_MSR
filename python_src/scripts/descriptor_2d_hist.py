#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

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

def compute_hist_2d(keypts_2d: np.ndarray, nbins: int = 8) -> (np.ndarray):
    '''Compute histogram based on number of detections in each angular sector around current keypoint

    Argument
    --------
    - keypts_2d: array of all 2d keypoints from the RGB frame of interest
    - nbins: number of bins to compute the histogram

    Returns
    -------
    - hist_2d: a histogram for each feature in the RGB-D frame of interest
    '''
    hist_2d = np.zeros((keypts_2d.shape[0], nbins))
    for i, kp in enumerate(keypts_2d):
        angles = np.arctan2(np.delete(keypts_2d, i, 0)[:, 1] - kp[1], np.delete(keypts_2d, i, 0)[:, 0] - kp[0])
        angles = np.array([wrapTo2Pi(-angle) for angle in angles]) * 90 / np.pi

        hist_2d[i] = np.histogram(angles, bins=nbins, range=(0, 360), density=False)[0]

    return hist_2d / keypts_2d.shape[0]

def compute_descriptor(keypoints_2d: np.ndarray, hist: np.ndarray) -> (np.ndarray):
    ''' Compute Descriptor for each 2d feature in the RGB frame
        descriptor = [2D location, histogram]
    '''
    descriptors = np.zeros((keypoints_2d.shape[0], 2 + hist.shape[1]))
    for i, kp in enumerate(keypoints_2d):
        descriptors[i] = np.r_[kp, hist[i]]
    return descriptors

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script generates custom descriptors for 2D macro keypoints''')

    parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/bboxes.pickle', type=str,
        help='Path to the object detection bounding box coordinates')
    parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations_rgbd.txt', type=str,
        help='Path to the associations file for RGB and Depth frames')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save_descriptor', default=False, type=bool, help='Save computed Descriptors as .npz file')
    args = parser.parse_args()

    bboxes_d, rgb_names = process_input_data(args.bboxes_path, args.associations_path)
     
    num_of_frames = len(rgb_names)
    skip_frames = 10
    for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):
        rgb_frame_1 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_names[n]), cv2.COLOR_RGB2BGR)
        keypts_2d_1 = np.array(bboxes_d[os.path.basename(rgb_names[n])])

        rgb_frame_2 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_names[n + skip_frames]), cv2.COLOR_RGB2BGR)
        keypts_2d_2 = np.array(bboxes_d[os.path.basename(rgb_names[n + skip_frames])])

        hist_1 = compute_hist_2d(keypts_2d_1)
        hist_2 = compute_hist_2d(keypts_2d_2)

        descriptors_1 = compute_descriptor(keypts_2d_1, hist_1)
        descriptors_2 = compute_descriptor(keypts_2d_2, hist_2)

        M = compute_matches(descriptors_1[:, 2:], descriptors_2[:, 2:], rgb_frame_1.shape[0], rgb_frame_1.shape[1])

        P1 = descriptors_1[:, :2].astype(int)
        P2 = descriptors_2[:, :2].astype(int)

        if(M.shape[0] > 0):
            H, M = compute_homography_ransac(P1, P2, M)

        w = rgb_frame_1.shape[1]
        rgb_match_frame = np.concatenate((rgb_frame_1, rgb_frame_2), 1)

        for kp in descriptors_1:
            cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 10, (255, 0, 0), 2)

        for kp in descriptors_2:
            cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                       int(kp[1])), 10, (0, 0, 255), 2)

        for kp_l, kp_r in zip(descriptors_1[M[:, 0]].astype(int), descriptors_2[M[:, 1]].astype(int)):
            cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)

        if args.visualize:
            cv2.imshow('matches', rgb_match_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.save_descriptor:
            cv2.imwrite(
                '../../eval_data/custom_2d_desc/{}.png'.format(n), rgb_match_frame)