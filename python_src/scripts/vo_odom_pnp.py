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
from utils.icp import *
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
    rot_quat = tf.Rotation.from_matrix(TF[:-1, :-1]).as_quat()
    return list(np.r_[t, origin, rot_quat])

def getMatches(data_img1: dict, data_img2: dict,threshold = 0.7):
    
    points_1 = data_img1['points'][:,:2]
    points_2 = data_img2['points'][:,:2]

    descriptor_1 = data_img1['desc']
    descriptor_2 = data_img2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T,threshold).astype(np.int64)

    return points_1[M[:,0]],points_2[M[:,1]]

def convert23d(sample: dict, keypoints: np.ndarray, k: np.ndarray):

    depth_factor = 1000
    permute = np.array([[0,1],[1,0]])
    permute_keypoints = keypoints @ permute
    depth_map = sample['depth'][permute_keypoints[:,0],permute_keypoints[:,1]] / depth_factor

    transformed_points = np.hstack((keypoints,np.ones((keypoints.shape[0],1))))
    ray_dir = (np.linalg.inv(k) @ transformed_points.T).T
    ray_dir =  ray_dir / ray_dir[:,-1].reshape(-1,1)

    points_3d = np.multiply(ray_dir,depth_map.reshape(-1,1))
   
    
    return points_3d

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='''Odometry using superpoint matching''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/apples_big_2021-10-14-all/', type=str,
                        help='Path to the root directory of the dataset')

    parser.add_argument('-n','--skip_frames',default = 1, help= "Number of frames to skip")

    args = parser.parse_args()

    if not "all" in args.data_root_path:
        dataset = DatasetOdometry(args.data_root_path)
    else:
        dataset = DatasetOdometryAll(args.data_root_path)
   
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    
    if "right" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    skip_frames = args.skip_frames
    # rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    # rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)
    
    viz = False
   
    trans = np.eye(4)
    progress_bar = tqdm(range(0,len(dataset)-skip_frames,skip_frames))
    x = []
    y = []
    z = []
    poses = []
    poses_inv = []

    for i in progress_bar:

        matches_img1, matches_img2 = getMatches(dataset[i],dataset[i+skip_frames])    
        
        if viz:

            w = 640
            points3d_1 = convert23d(dataset[i],matches_img1.astype(np.int64),rgb_camera_intrinsic.intrinsic_matrix)
            points3d_2 = convert23d(dataset[i+skip_frames],matches_img2.astype(np.int64),rgb_camera_intrinsic.intrinsic_matrix)
            idx = np.where((points3d_1[:,-1] <= 5) & (points3d_1[:,-1] > 0) & (points3d_2[:,-1] <= 5) & (points3d_2[:,-1] > 0))[0]
        
            rgb_match_frame = np.concatenate((dataset[i]['rgb'], dataset[i+1]['rgb']), 1)
            for kp in matches_img1[idx]:
                cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

            for kp in matches_img2[idx]:
                cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                        int(kp[1])), 3, (0, 0, 255), -1)
            output_frame = np.copy(rgb_match_frame)
        
            for kp_l, kp_r in zip(matches_img1.astype(np.int64), matches_img2.astype(np.int64)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            cv2.imshow("output", output_frame)

            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        points3d_1 = convert23d(dataset[i],matches_img1.astype(np.int64),rgb_camera_intrinsic.intrinsic_matrix)
        points3d_2 = convert23d(dataset[i+skip_frames],matches_img2.astype(np.int64),rgb_camera_intrinsic.intrinsic_matrix)
        idx = np.where((points3d_1[:,-1] <= 10) & (points3d_1[:,-1] > 0) & (points3d_2[:,-1] <= 10) & (points3d_2[:,-1] > 0))[0]
        # print(type(cv2.Rodrigues(np.eye(3))[0]))
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points3d_1[idx], matches_img2[idx], rgb_camera_intrinsic.intrinsic_matrix.astype(np.double), None)
        # rvec, tvec = cv2.solvePnPRefineLM(points3d_1[idx], matches_img2[idx], rgb_camera_intrinsic.intrinsic_matrix.astype(np.double), None, rvec,tvec,criteria)
        rmat = cv2.Rodrigues(rvec)[0]

        T_out = np.eye(4)
        T_out[:-1,:-1] = rmat
        T_out[:-1,-1] = tvec.reshape((3,))
        
        trans =  trans @ np.linalg.inv(T_out)
        # trans_inv = np.linalg.inv(trans)
        poses.append(odom_from_SE3(dataset[i]['timestamp'],trans))
        # poses_inv.append(odom_from_SE3(dataset[i]['timestamp'],trans_inv))

        x.append(trans[:-1,-1][0])
        y.append(trans[:-1,-1][1])
        z.append(trans[:-1,-1][2])
    
    # np.savetxt("x_pnp.txt",x)
    # np.savetxt("y_pnp.txt",y)
    # np.savetxt("z_pnp.txt",z)
    np.savetxt("poses_all.txt",poses)
    # np.savetxt("poses_inv_freiburg.txt",poses_inv)

    x= np.asarray(x)
    y= np.asarray(y)
    z= np.asarray(z)

    gt = np.loadtxt("../../datasets/phenorob/groundtruth.txt")
    gtx = gt[:,1]
    gty = gt[:,2]
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.plot(x, y, 'r')
    ax1.plot(gtx,gty,'b')
    ax2.plot(z, y, 'r')
    ax2.plot(gtx,gty,'b')
    ax3.plot(z, x, 'r')
    ax3.plot(gtx,gty,'b')
    plt.show()


    # T_ = icp_known_corresp(points3d_1.T,points3d_2.T,idx,idx)

        # import ipdb;ipdb.set_trace()
        # # print(type(points_3d))
        # # import ipdb;ipdb.set_trace()
        # rgbd1 = create_rgbdimg(dataset[i]['rgb'],dataset[i]['depth'])
        # source_pcd = RGBD2PCL(rgbd1,rgb_camera_intrinsic,False)
        # pcl1 = o3d.geometry.PointCloud()
        # pcl1.points = o3d.utility.Vector3dVector(points3d_1)
        # pcl1.paint_uniform_color([0, 0, 1])
        # pcl2 = o3d.geometry.PointCloud()
        # pcl2.points = o3d.utility.Vector3dVector(points3d_2)
        # pcl2.paint_uniform_color([1, 0, 0])
        
        # result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(pcl2,pcl1,o3d.utility.Vector2iVector(np.dstack((idx,idx))[0]))
        
        # trans = trans.dot(result.transformation)