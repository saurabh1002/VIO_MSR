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

    ROOT_DIR = '../../eval_data/'

    cam_traj = utils.read_file(ROOT_DIR + "rgbd_orb3/CameraTrajectory.txt")

    keyframe_traj = utils.read_file(ROOT_DIR + "rgbd_orb3/KeyFrameTrajectory.txt")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.plot(keyframe_traj[:, 3], keyframe_traj[:, 1], 'r', label='keyframe trajectory')
    ax.plot(cam_traj[:, 1], cam_traj[:, 2], cam_traj[:, 3], 'g', label='camera trajectory')
    # plt.scatter(keyframe_traj[0, 3], keyframe_traj[0, 1], 50, 'r', 'x')
    # ax.scatter(cam_traj[0, 1], cam_traj[0, 2], cam_traj[0, 3], 50, 'g', 'x')
    plt.legend()
    plt.show()