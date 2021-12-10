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
import numpy as np
from matplotlib import pyplot as plt

def display_imgs(im_dict: dict, wait_val: int = 0, new_window: bool = False):
    """Helper function for displaying images"""
    for window_name, img in im_dict.items():
        cv2.imshow(window_name, img)
    cv2.waitKey(wait_val)
    if new_window:
        cv2.destroyAllWindows()


img_dataset_path = '../../datasets/phenorob/images_apples/rgb_for_vocab/'

orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, fastThreshold=5)

NUM_IMAGES = 1579

num_of_keypoints = np.zeros(NUM_IMAGES)
num_of_matches = np.zeros(NUM_IMAGES)

step = 100

for i in range(1, NUM_IMAGES):
    img_rgb_a = cv2.imread(img_dataset_path + 'image{}.png'.format(i))
    img_bgr_a = cv2.cvtColor(img_rgb_a, cv2.COLOR_RGB2BGR)

    h, w, d = img_bgr_a.shape

    img_rgb_b = cv2.imread(img_dataset_path + 'image{}.png'.format(i + step))
    img_bgr_b = cv2.cvtColor(img_rgb_b, cv2.COLOR_RGB2BGR)
    
    keypoints_a, descriptors_a = orb.detectAndCompute(img_bgr_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(img_bgr_b, None)
    
    num_of_keypoints[i] = len(keypoints_a)

    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
    matches = matcher.match(descriptors_a, descriptors_b)

    num_of_matches[i] = len(matches)

    matches = sorted(matches, key=lambda x:x.distance)

    img_match = np.zeros((h, 2 * w, d), np.uint8)

    cv2.drawMatches(img_bgr_a, keypoints_a, img_bgr_b, keypoints_b, matches[:50], img_match, flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.putText(img_match, 'Frame i: No. of keypoints detected: {}'.format(len(keypoints_a)), (0, 450), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img_match, 'Frame i+{}: No. of keypoints detected: {}'.format(step, len(keypoints_b)), (640, 450), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)

    cv2.imshow('matches', img_match)
    cv2.waitKey(0)
    # cv2.imwrite('tmp/img_{}.png'.format(i), img_match)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(num_of_keypoints)
ax1.set_title("Number of Keypoints")
ax2.hist(num_of_matches)
ax2.set_title("Number of Matches")
plt.waitforbuttonpress()
plt.close()