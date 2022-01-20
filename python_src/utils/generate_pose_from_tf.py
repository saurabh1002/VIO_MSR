#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from ctypes import util
import scipy.spatial.transform as tf
from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.append('..')

import utils

def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:3, :3]).as_quat()
    return list(np.r_[t, origin, rot_quat])

if __name__ == "__main__":
    ROOT_DIR = '../../datasets/phenorob/images_apples_right/'

    gt_tf = utils.read_file(ROOT_DIR + "poses-ali-sam_apples-big_14-10.txt")

    gt_poses = []

    for tf16 in gt_tf:
        gt_poses.append(odom_from_SE3(tf16[0], tf16[1:].reshape(4, 4)))

    np.savetxt(ROOT_DIR + 'groundtruth.txt', np.array(gt_poses))
