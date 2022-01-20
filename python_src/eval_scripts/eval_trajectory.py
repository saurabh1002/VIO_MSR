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

    ROOT_DIR = '../../results/'

    cam_traj = utils.read_file(ROOT_DIR + "poses_model_to_frame.txt")
    
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.set_title("XY - plane")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.plot(cam_traj[0, 1], cam_traj[0, 2], 'rx', label='Starting Point')
    ax.plot(cam_traj[:, 1], cam_traj[:, 2], 'g', label='Trajectory')
    ax.legend()

    ax = fig.add_subplot(312)
    ax.set_title("XZ - plane")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.plot(cam_traj[0, 1], cam_traj[0, 3], 'rx', label='Starting Point')
    ax.plot(cam_traj[:, 1], cam_traj[:, 3], 'g', label='Trajectory')
    ax.legend()

    ax = fig.add_subplot(313)
    ax.set_title("YZ - plane")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.plot(cam_traj[0, 2], cam_traj[0, 3], 'rx', label='Starting Point')
    ax.plot(cam_traj[:, 2], cam_traj[:, 3], 'g', label='Trajectory')
    ax.legend()
    plt.show()