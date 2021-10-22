#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


from typing import NoReturn
from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.append('..')

from utils import utils

if __name__ == "__main__":
    # gt_traj = utils.read_file("../../../../Datasets/rgbd_dataset_freiburg1_xyz/groundtruth.txt")

    cam_traj = utils.read_file("../../../ORB_SLAM2/outputs/apple_cam_traj.txt")

    keyframe_traj = utils.read_file("../../../ORB_SLAM2/outputs/apple_KeyFrameTrajectory.txt")
    
    plt.plot(keyframe_traj[:, 1], keyframe_traj[:, 2], 'r')
    # plt.plot(cam_traj[:, 1], cam_traj[:, 2], 'g')
    plt.show()