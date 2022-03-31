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
import scipy.spatial
from tqdm import tqdm
import numpy.linalg as la

import sys
sys.path.append('..')

from utils.utils import wrapTo2Pi

def find_k_nearest(keypoints: np.ndarray, k: int = 4):
    dist = scipy.spatial.distance.cdist(keypoints, keypoints)
    sort_ids = np.argsort(dist)

    return sort_ids[:, 1:k+1]

def compute_descriptor(keypoint: np.ndarray, k_neigbors: np.ndarray):
    k = k_neigbors.shape[0]
    dist = la.norm(keypoint - k_neigbors, 2, 1)
    
    dist_ratios = dist[:-1] / dist[-1]
    angles = np.arctan2(k_neigbors[:-1, 1] - keypoint[1], k_neigbors[:-1, 0] - keypoint[0]) - np.arctan2(k_neigbors[-1, 1] - keypoint[1], k_neigbors[-1, 0] - keypoint[0]) 
    angles = np.array([wrapTo2Pi(-angle) for angle in angles]) / (2 * np.pi)

    sort_idx = np.argsort(angles)

    desc = np.r_[keypoint, dist_ratios[sort_idx], angles[sort_idx]]
    return desc, sort_idx


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''
    This script takes the keypoints kdtree object and generates custom descriptors for each keypoint
    ''')
    parser.add_argument('--bboxes_path', default='../../datasets/MineApple/bboxes.pickle', type=str,
     help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('--k', default=4, type=int, help='Number of nearest neighbors to use for descriptor')
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save_descriptor', default=False, type=bool, help='Save computed Descriptors as .npz file')
    args = parser.parse_args()

    with open(args.bboxes_path, "rb") as f:
        bboxes_data = pickle.load(f)

    image_paths = bboxes_data.keys()
    num_of_images = len(image_paths)

    descriptors = {}

    for i, image in enumerate(image_paths):
        print('Computing Keypoint Descriptors for image {}/{}.....\n'.format(i + 1, num_of_images))
        
        image_keypoints = np.asarray(bboxes_data[image])
        image_keypoints = image_keypoints[:, 4:]
        
        if image_keypoints.shape[0] >= args.k + 1:
            neighbor_ids = find_k_nearest(image_keypoints, args.k)
            
            desc = np.zeros((image_keypoints.shape[0], args.k * 2))
            for i in tqdm(range(image_keypoints.shape[0])):  
                desc[i], sort_idx = compute_descriptor(image_keypoints[i], image_keypoints[neighbor_ids[i]])
                if args.visualize:
                    delay = 100
                    cv2.namedWindow(image, cv2.WINDOW_NORMAL)
                    img = cv2.imread('../../datasets/phenorob/mapping/images_apples_right/rgb/' + image)
                    cv2.circle(img, (int(image_keypoints[i, 0]), int(image_keypoints[i, 1])), 10, (0, 0, 255), 3)
                    cv2.imshow(image, img)
                    cv2.waitKey(delay)
                    for j in range(args.k):
                        cv2.circle(img, (int(image_keypoints[neighbor_ids[i, j], 0]), int(image_keypoints[neighbor_ids[i, j], 1])), 10, (255, 0, 0), 3)
                        cv2.imshow(image, img)
                        cv2.waitKey(delay)
                    cv2.line(img, (int(image_keypoints[i, 0]), int(image_keypoints[i, 1])), 
                        (int(image_keypoints[neighbor_ids[i, -1], 0]), int(image_keypoints[neighbor_ids[i, -1], 1])), (0, 255, 0), 2)
                    cv2.imshow(image, img)
                    cv2.waitKey(delay)
                    for id in range(args.k - 1):
                        cv2.line(img, (int(image_keypoints[i, 0]), int(image_keypoints[i, 1])), 
                            (int(image_keypoints[neighbor_ids[i, sort_idx[id]], 0]), int(image_keypoints[neighbor_ids[i, sort_idx[id]], 1])),
                            (255, 0, 255), 2)
                        cv2.imshow(image, img)
                        cv2.waitKey(delay)
            cv2.destroyAllWindows()
            descriptors[image] = desc
        else:
            print("[WARNING] : Not enough features to compute descriptors for image {}/{}\n".format(i + 1, num_of_images))

    if args.save_descriptor:
        np.save('keypoint_descriptors_{}'.format(args.k), descriptors)
        print('Saved keypoint descriptors in `keypoints_descriptors_{}.npy` file\n'.format(args.k))