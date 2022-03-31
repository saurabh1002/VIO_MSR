import os
import sys
sys.path.append(os.pardir)
import pickle
import argparse
from tqdm import tqdm
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_input_data(bboxes_path: str, associations_path: str) ->Tuple[dict, list]:
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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''This script generates custom descriptors for 3D macro keypoints''')

    # Dataset paths
    parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/bboxes.pickle', type=str,
        help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations_rgbd.txt', type=str,
        help='Path to the associations file for RGB and Depth frames')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
    args = parser.parse_args()

    bboxes_d, rgb_names = process_input_data(args.bboxes_path, args.associations_path)

    num_of_frames = len(rgb_names)

    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, fastThreshold=5)

    num_of_keypoints = np.zeros(num_of_frames)
    num_of_matches = np.zeros(num_of_frames)

    skip_frames = 5
    for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):
        img_rgb_a = cv2.imread(args.data_root_path + rgb_names[n])
        img_bgr_a = cv2.cvtColor(img_rgb_a, cv2.COLOR_RGB2BGR)

        h, w, d = img_bgr_a.shape

        img_rgb_b = cv2.imread(args.data_root_path + rgb_names[n + skip_frames])
        img_bgr_b = cv2.cvtColor(img_rgb_b, cv2.COLOR_RGB2BGR)
        
        keypoints_a, descriptors_a = orb.detectAndCompute(img_bgr_a, None)
        keypoints_b, descriptors_b = orb.detectAndCompute(img_bgr_b, None)
        
        num_of_keypoints[n] = len(keypoints_a)

        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
        matches = matcher.match(descriptors_a, descriptors_b)

        num_of_matches[n] = len(matches)

        matches = sorted(matches, key=lambda x:x.distance)

        img_match = np.zeros((h, 2 * w, d), np.uint8)

        cv2.drawMatches(img_bgr_a, keypoints_a, img_bgr_b, keypoints_b, matches[:50], img_match, 2, (0, 255, 255), (255, 0, 0), None, flags=cv2.DrawMatchesFlags_DEFAULT)
        cv2.putText(img_match, 'Frame i: No. of keypoints detected: {}'.format(len(keypoints_a)), (0, 450), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(img_match, 'Frame i+{}: No. of keypoints detected: {}'.format(skip_frames, len(keypoints_b)), (640, 450), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)

        cv2.imshow('matches', img_match)
        cv2.waitKey(0)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(num_of_keypoints)
    ax1.set_title("Number of Keypoints")
    ax2.hist(num_of_matches)
    ax2.set_title("Number of Matches")
    plt.waitforbuttonpress()
    plt.close()