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
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import numpy as np

import os
import sys
sys.path.append(os.pardir)


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
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
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
    n_clusters = 2
    np.random.seed(1)
    label_color = np.random.randint(0, 255, (n_clusters, 3), np.int16)
    for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):
        rgb_frame_1 = cv2.cvtColor(cv2.imread(args.data_root_path + rgb_frame_names[n]), cv2.COLOR_RGB2BGR)

        points_1 = points_all[rgb_frame_names[n].split('/')[1]][:, :2]

        descriptors_1 = descriptors_all[rgb_frame_names[n].split('/')[1]]

        kmeans_1 = KMeans(n_clusters=n_clusters, random_state=0).fit(descriptors_1)

        labels_1 = kmeans_1.labels_

        if args.visualize or args.save:
            w = rgb_frame_1.shape[1]
            for i, kp in enumerate(points_1):
                color = label_color[labels_1[i]].tolist()
                cv2.circle(rgb_frame_1, (int(kp[0]), int(kp[1])), 4, tuple(color), -1)

            if args.visualize:
                cv2.namedWindow('Clusters')
                cv2.imshow('Clusters', rgb_frame_1)
                cv2.waitKey(0)

    cv2.destroyAllWindows()