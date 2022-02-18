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
from utils.dataloader import DatasetOdometry
from utils.utils import *
from superpoint_matching import nn_match_two_way
import scipy.spatial.transform as tf
import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
from pointtracker import PointTracker
from typing import Tuple
import cv2
import matplotlib.pyplot as plt


def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:3, :3]).as_quat()
    return list(np.r_[t, origin, rot_quat])

def getRnt(points_1: np.ndarray, points_2: np.ndarray, camera_intrinsics: np.ndarray):
    
    prob = 0.999
    threshold = 0.00001
    method = cv2.RANSAC 
    
    e, mask = cv2.findEssentialMat(points_1,
                                   points_2,
                                   camera_intrinsics.astype(np.double),
                                   method=method, 
                                   prob=prob, 
                                   threshold=threshold)
    # print(points_1.shape)
    condition = mask == 1
    condition = condition.reshape((points_1.shape[0],))
    
    _,R,t,_ = cv2.recoverPose(e, points_1[condition], points_2[condition])
   
    return R,t.reshape(3,)

def getMatches(data_img1: dict, data_img2: dict,threshold = 0.3):
    
    points_1 = data_img1['points'][:,:2]
    points_2 = data_img2['points'][:,:2]

    descriptor_1 = data_img1['desc']
    descriptor_2 = data_img2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T,threshold).astype(np.int64)

    return points_1[M[:,0]],points_2[M[:,1]]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='''Odometry using superpoint matching''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/apples_big_2021-10-14-14-51-44_1/', type=str,
                        help='Path to the root directory of the dataset')

    parser.add_argument('-n','--skip_frames',default=5, help="Number of frames to skip")

    args = parser.parse_args()
    dataset = DatasetOdometry(args.data_root_path)
   
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_mat = np.asarray(rgb_camera_intrinsic.intrinsic_matrix)
    if "right" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    
    skip_frames = args.skip_frames
    
    I = np.eye(4)
    trans_init = np.eye(4)
    trans_abs = np.eye(4)

    progress_bar = tqdm(range(0,len(dataset)-skip_frames,skip_frames))

    init = False
    first_iter = False
    rel_scale = 0.0
    

    current_t = trans_init[:-1,-1]
    current_R = trans_init[:-1,:-1]
    current_pcd = None
  
    for i in progress_bar:
        # First time calculate with ICP and essential matrix both to fix the scale
        
        matches_img1, matches_img2 = getMatches(dataset[i],dataset[i+1])
        R,t = getRnt(matches_img1,matches_img2,np.asarray(rgb_camera_intrinsic.intrinsic_matrix))
        rgbd1 = create_rgbdimg(dataset[i]['rgb'],dataset[i]['depth'])
        rgbd2 = create_rgbdimg(dataset[i+skip_frames]['rgb'],dataset[i+skip_frames]['depth'])

        target_pcd = RGBD2PCL(rgbd1,rgb_camera_intrinsic,True)
        source_pcd = RGBD2PCL(rgbd2,rgb_camera_intrinsic,False)
        
        if not init:

            rgbd1 = create_rgbdimg(dataset[i]['rgb'],dataset[i]['depth'])
            rgbd2 = create_rgbdimg(dataset[i+skip_frames]['rgb'],dataset[i+skip_frames]['depth'])

            target_pcd = RGBD2PCL(rgbd1,rgb_camera_intrinsic,True)
            source_pcd = RGBD2PCL(rgbd2,rgb_camera_intrinsic,False)

            result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                0.05,
                I,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )
            t_abs = result.transformation[:-1,-1]
            rel_scale = np.linalg.norm(t_abs) / np.linalg.norm(t)

            init = False

        
        if not first_iter:
            trans_init[:-1,:-1] = R
            trans_init[:-1,-1] = (rel_scale * t)
            
            current_pcd = target_pcd + source_pcd.transform(trans_init)
            first_iter = True
        
        else:
            trans_init[:-1,:-1] = R
            trans_init[:-1,-1] = (rel_scale * t)
            current_pcd = current_pcd +  source_pcd.transform(trans_init)

    
            o3d.visualization.draw_geometries([target_pcd,source_pcd.transform(trans_init)])


        

        



        