import argparse
import pickle
import os
from typing import Dict

from tqdm import tqdm

import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d


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

parser = argparse.ArgumentParser(description='''This script generates custom descriptors for 3D macro keypoints''')

# Dataset paths
parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/detection.pickle', type=str,
    help='Path to the centernet object detection bounding box coordinates')
parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations.txt', type=str,
    help='Path to the associations file for RGB and Depth frames')
parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
    help='Path to the root directory of the dataset')

parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
args = parser.parse_args()

bboxes_d, rgb_names, depth_names = process_input_data(args.bboxes_path, args.associations_path)
num_of_frames = len(rgb_names)

rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)

feature_params = { 'maxCorners' : 100,
                       'qualityLevel' : 0.3,
                       'minDistance' : 7,
                       'blockSize' : 7}

lk_params = {'winSize': (15, 15),
                  'maxLevel': 2,
                  'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

color = np.random.randint(0, 255, (100, 3))
        

skip_frames = 5
for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):

    img_rgb_1 = cv2.imread(args.data_root_path + rgb_names[n])
    img_depth_1 = cv2.imread(args.data_root_path + depth_names[n], cv2.CV_16UC1)

    img_rgb_2 = cv2.imread(args.data_root_path + rgb_names[n + skip_frames])
    img_depth_2 = cv2.imread(args.data_root_path + depth_names[n + skip_frames], cv2.CV_16UC1)

    img_gray_1 = cv2.cvtColor(img_rgb_1, cv2.COLOR_RGB2GRAY)
    img_gray_2 = cv2.cvtColor(img_rgb_2, cv2.COLOR_RGB2GRAY)

    # p1 = cv2.goodFeaturesToTrack(img_gray_1, mask = None, **feature_params)

    # mask = np.zeros_like(img_rgb_1)

    # p2, st, err = cv2.calcOpticalFlowPyrLK(img_gray_1, img_gray_2, p1, None, **lk_params)

    # if p2 is not None:
    #     good_1 = p1[st == 1]
    #     good_2 = p2[st == 1]

    # for i, (new, old) in enumerate(zip(good_2, good_1)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    #     img_rgb_2 = cv2.circle(img_rgb_2, (int(a), int(b)), 5, color[i].tolist(), -1)

    # img = cv2.add(img_rgb_2, mask)
    # cv2.imshow('flow', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    flow = cv2.calcOpticalFlowFarneback(img_gray_1, img_gray_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros_like(img_rgb_1)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 90 / np.pi
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('dense flow', bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()