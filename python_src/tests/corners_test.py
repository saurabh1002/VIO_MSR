#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_imgs(im_dict: dict, wait_val: int = 0, new_window: bool = False):
    """Helper function for displaying images"""
    for window_name, img in im_dict.items():
        cv2.imshow(window_name, img)
    cv2.waitKey(wait_val)
    if new_window:
        cv2.destroyAllWindows()


img_dataset_path = '../../datasets/phenorob/mapping/images_apples/rgb_for_vocab/'

fast = cv2.FastFeatureDetector_create()
fast.setThreshold(50)

# Print all default params
print("User parameters for the FAST feature detector")
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())

for i in range(1000):
    img_rgb = cv2.imread(img_dataset_path + 'image{}.png'.format(i + 1))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    keypoints = fast.detect(img_bgr, None)

    img_corners = np.zeros_like(img_bgr)
    cv2.drawKeypoints(img_bgr, keypoints, img_corners, color=(255,0,0))

    display_imgs({'Fast corners': img_corners}, wait_val=0)

