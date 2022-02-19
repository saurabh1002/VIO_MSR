from calendar import c
import dis
import os
from random import sample
from sre_constants import SUCCESS
import sys
from click import progressbar
from cv2 import SOLVEPNP_ITERATIVE, threshold
from pandas import concat
from sklearn.preprocessing import scale

from zmq import device
sys.path.append(os.pardir)
from utils.dataloader import DatasetOdometry, DatasetOdometryAll
from utils.utils import *
from superpoint_matching import nn_match_two_way, process_superpoint_feature_descriptors
import scipy.spatial.transform as tf
import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
from pointtracker import PointTracker
from typing import Tuple
import cv2
import ipdb
import matplotlib.pyplot as plt


def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    # origin = [vec[2],-vec[0],-vec[1]]
    rot_quat = tf.Rotation.from_matrix(TF[:-1, :-1]).as_quat()
    return list(np.r_[t, origin, rot_quat])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='''Odometry using superpoint matching''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/apples_big_2021-10-14-all/', type=str,
                        help='Path to the root directory of the dataset')

    parser.add_argument('-n','--skip_frames',default=1, help="Number of frames to skip")

    args = parser.parse_args()
    if not "all" in args.data_root_path:
        dataset = DatasetOdometry(args.data_root_path)
    else:
        dataset = DatasetOdometryAll(args.data_root_path)
   
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_mat = np.asarray(rgb_camera_intrinsic.intrinsic_matrix)
    
    if "right" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    
    skip_frames = args.skip_frames
    
    I = np.eye(4)
    
    trans = np.eye(4)
    progress_bar = tqdm(range(0,len(dataset)-skip_frames,skip_frames))

    
    poses = []
    x = []
    y = []
    z = []
    
    current_pcd = None

    poses.append(odom_from_SE3(dataset[0]['timestamp'],trans))
    diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    
    for i in progress_bar:
        # First time calculate with ICP and essential matrix both to fix the scale
        
  
        rgbd1 = create_rgbdimg(dataset[i]['rgb'],dataset[i]['depth'])
        rgbd2 = create_rgbdimg(dataset[i+skip_frames]['rgb'],dataset[i+skip_frames]['depth'])

        source_pcd = RGBD2PCL(rgbd1,rgb_camera_intrinsic,False)
        target_pcd = RGBD2PCL(rgbd2,rgb_camera_intrinsic,True)

        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.05,
            I,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
           
        trans = trans @ result.transformation
        x.append(trans[:-1,-1][0])
        y.append(trans[:-1,-1][1])
        z.append(trans[:-1,-1][2])
   
    np.savetxt("x_icp.txt",np.asarray(x))
    np.savetxt("y_icp.txt",np.asarray(y))
    np.savetxt("z_icp.txt",np.asarray(z))
    fig,ax1 = plt.subplots()
    ax1.plot(x, y, 'r')
   
    
    plt.show()

    # np.savetxt(args.data_root_path + "pose.txt", np.array(poses))
        



        