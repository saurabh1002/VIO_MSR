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
import matplotlib.pyplot as plt


def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    # origin = [vec[2],-vec[0],-vec[1]]
    rot_quat = tf.Rotation.from_matrix(TF[:-1, :-1]).as_quat()
    return list(np.r_[t, origin, rot_quat])




def getRnt(points_1: np.ndarray, points_2: np.ndarray, camera_intrinsics: np.ndarray):
     
    prob = 0.999
    threshold = 1
    method = cv2.RANSAC 
    
    # F, mask = cv2.findFundamentalMat(points_1,
    #                                points_2,
    #                                method=method, 
    #                                ransacReprojThreshold=threshold,
    #                                confidence=prob)

    # F,mask = computeFundMat(points_1,points_2)

    k_inv = np.linalg.inv(camera_intrinsics)
    # e = k_inv.T @ F @ k_inv

    # transformed_points_1 = toEuclidean(toHomogenous(points_1) @ k_inv)
    # transformed_points_2 = toEuclidean(toHomogenous(points_2) @ k_inv)
    # import ipdb;ipdb.set_trace()
    
    e, mask = cv2.findEssentialMat(points_1,
                                   points_2,
                                   k_inv.astype(np.double),
                                   method=method, 
                                   prob=prob, 
                                   threshold=threshold)
    
    U,S,V = np.linalg.svd(e)
    # import ipdb;ipdb.set_trace()
    condition = mask == 1
    condition = condition.reshape((points_1.shape[0],))

    _,R,t,_= cv2.recoverPose(e, points_1[mask], points_2[mask],camera_intrinsics.astype(np.double))

    # print(mask1.shape)
    # pts = (R @ np.hstack((points_1,np.ones((points_1.shape[0],1)))).T + t).T
    # err = np.linalg.norm(pts[:,:-1]-points_2) / (points_2.shape[0])

    

    return R,t.reshape((3,))

def computeFundMat(corr1, corr2):

    fundMat = np.zeros((3,3))
    mean1 = np.mean(corr1, 0, dtype=float) # (2, )
    mean2 = np.mean(corr2, 0, dtype=float) # (2, )

    rmsd1 = np.mean(np.sqrt(np.power(corr1[:, 0] - mean1[0], 2) + np.power(corr1[:, 1] - mean1[1], 2)))
    rmsd2 = np.mean(np.sqrt(np.power(corr2[:, 0] - mean2[0], 2) + np.power(corr2[:, 1] - mean2[1], 2)))


    ncorr1 = (corr1 - mean1) * np.sqrt(2) / rmsd1
    ncorr2 = (corr2 - mean2) * np.sqrt(2) / rmsd2


    prob = 0.999
    threshold = 1
    method = cv2.FM_RANSAC 
   
    Tl = np.array([np.sqrt(2) / rmsd1, 0, -1 * mean1[0] * np.sqrt(2) / rmsd1, 0, np.sqrt(2) / rmsd1,
                   -1 * mean1[1] * np.sqrt(2) / rmsd1, 0, 0, 1]).reshape((3,3))
    Tr = np.array([np.sqrt(2) / rmsd2, 0, -1 * mean2[0] * np.sqrt(2) / rmsd2, 0, np.sqrt(2) / rmsd2,
          -1 * mean2[1] * np.sqrt(2) / rmsd2, 0, 0, 1]).reshape((3,3))
    F = np.dot(Tr.transpose(), np.dot(F, Tl))
    fundMat = F

    return fundMat,mask

def getRnTdepth(points_1: np.ndarray, points_2: np.ndarray, camera_intrinsics: np.ndarray, img_depth_1: np.ndarray, img_depth_2: np.ndarray):
    
    cx = camera_intrinsics[0,2]
    cy = camera_intrinsics[1,2]
    fx = camera_intrinsics[0,0]
    fy = camera_intrinsics[1,1]
    max_depth = 3000
    depth_mask = []
    keypoint_1 = np.empty((0,3))

    for i, (u,v) in enumerate(points_1):
        z = img_depth_1[int(v),int(u)]

        if z > max_depth:
            depth_mask.append(i)
            continue
        
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        keypoint_1 = np.vstack((keypoint_1,np.array([x,y,z])))

    _, rvec, tvec, inliers = cv2.solvePnPRansac(keypoint_1, points_2[depth_mask], camera_intrinsics.astype(np.double), None)
    rmat = cv2.Rodrigues(rvec)[0]

    return rmat,tvec



def getMatches(data_img1: dict, data_img2: dict,threshold = 0.7):
    
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
    parser.add_argument('-i', '--data_root_path', default='/home/dhagash/Projects/SuperPointPretrainedNetwork/rgbd_dataset_freiburg1_desk/', type=str,
                        help='Path to the root directory of the dataset')

    parser.add_argument('-n','--skip_frames',default=1, help="Number of frames to skip")

    args = parser.parse_args()
    if not "all" in args.data_root_path:
        dataset = DatasetOdometry(args.data_root_path)
    else:
        dataset = DatasetOdometryAll(args.data_root_path)
   
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    
    skip_frames = args.skip_frames
    
   
    trans = np.eye(4)

    trans_res = np.eye(4)
    progress_bar = tqdm(range(0,len(dataset)-skip_frames,skip_frames))

    viz = False
    poses = []
    x = []
    y = []
    z = []
    
    
    print(f"K Matrix:\n{rgb_camera_intrinsic.intrinsic_matrix}")
    poses.append(odom_from_SE3(dataset[0]['timestamp'],trans))
   
    for i in progress_bar:
        # First time calculate with ICP and essential matrix both to fix the scale
        matches_img1, matches_img2 = getMatches(dataset[i],dataset[i+1])

        # idx =  np.where(np.linalg.norm(matches_img1-matches_img2,axis=1) > 1.3) [0]
        
        # print(np.linalg.norm(matches_img1-matches_img2))
        
        # R,t = getRnTdepth(matches_img1,matches_img2,np.asarray(rgb_camera_intrinsic.intrinsic_matrix), dataset[i]['depth'], dataset[i+1]['depth'])
        R,t = getRnt(matches_img1,matches_img2,np.asarray(rgb_camera_intrinsic.intrinsic_matrix))
        # print(R,t,error)
    
        # ipdb.set_trace()
        if viz:

            w = 640
            rgb_match_frame = np.concatenate((dataset[i]['rgb'], dataset[i+1]['rgb']), 1)
            for kp in matches_img1:
                cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

            for kp in matches_img2:
                cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                        int(kp[1])), 3, (0, 0, 255), -1)
            output_frame = np.copy(rgb_match_frame)
        
            for kp_l, kp_r in zip(matches_img1.astype(np.int64), matches_img2.astype(np.int64)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            cv2.imshow("output", output_frame)

            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
            

        trans_res[:-1,:-1] = R
        trans_res[:-1,-1] = np.array([1,0,0])
        trans = trans.dot(trans_res)
        x.append(trans[:-1,-1][0])
        y.append(trans[:-1,-1][1])
        z.append(trans[:-1,-1][2])
    


    fig,(ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(x, y, 'r')
    ax2.plot(y, z, 'r')
    ax3.plot(x, z, 'r')
    
    plt.show()

   


        