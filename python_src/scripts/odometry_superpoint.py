from calendar import c
import dis
import os
from random import sample
import sys

from zmq import device
sys.path.append(os.pardir)
from utils.dataloader import DatasetOdometry
from superpoint_matching import nn_match_two_way
import scipy.spatial.transform as tf
import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
from pointtracker import PointTracker
import cv2


def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:3, :3]).as_quat()
    return list(np.r_[t, origin, rot_quat])


def obtain_relative_scale(data,camera_mat):
    relative_scales = []

    trans = np.eye(4)
    tracker = PointTracker(2,0.2)
    for i in range(0,1):
        # 
        trans1 = np.eye(4)
        trans2 = np.eye(4)
        sample_1 = data[i]
        sample_2 = data[i+1]
        sample_3 = data[i+20]
        tracker.update(sample_1['points'].T,sample_1['desc'].T)
        # tracker.update(sample_2['points'].T,sample_2['desc'].T)
        tracker.update(sample_3['points'].T,sample_3['desc'].T)
        
        img = sample_1['rgb']
        tracks = tracker.get_tracks(2)
        out1 = img.astype('uint8')
        # print(out1.shape)
        tracks[:, 1] /= float(0.2)  # Normalize track scores to [0,1].
        tracker.draw_tracks(out1, tracks)
        cv2.imshow("dadaS",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # offsets =tracker.get_offsets()
        # print(offsets)
        # print(tracks[0])
        
        # print(tracks[np.where(tracks[:,1]<0.4)][0])

        # img2 =sample_2['rgb']
        # img3 = sample_3['rgb']
        # w = img1.shape[1]
        # w2 = img1.shape[1] + img2.shape[1]

        # rgb_match_frame = np.concatenate((img1, img2,img3), 1)
        # pt1 = (int(sample_1['points'][0][0]), int(sample_1['points'][0][1]))
        # pt2 = (int(sample_2['points'][436][0]) + w, int(sample_2['points'][436][1]))
        # pt3 = (int(sample_3['points'][881][0]) + w2 + w, int(sample_3['points'][881][1]))
         
        # cv2.line(rgb_match_frame,pt1,pt2, (0, 255, 255), 2)
        # cv2.line(rgb_match_frame,pt2,pt3, (0, 0, 255), 2)

        # M1 = nn_match_two_way(sample_1['desc'].T, sample_2['desc'].T,0.2)
        # M1 = M1.astype(np.int32)
        # e1,mask1=cv2.findEssentialMat(sample_1['points'][M1[:,0]],sample_2['points'][M1[:,1]],camera_mat.astype(np.double),method=cv2.RANSAC, prob=0.999, threshold=0.001)
        # condition1 = mask1 == 1
        # condition1 = condition1.reshape((M1.shape[0],))
        # _,R1,t1,_ = cv2.recoverPose(e1,sample_1['points'][M1[condition1,0]],sample_2['points'][M1[condition1,1]])
        # trans1[:-1,:-1] = R1
        # trans1[:-1,-1] = t1.reshape((3,))
        # points3d_1 = cv2.triangulatePoints(trans[:-1,:],trans1[:-1,:],sample_1['points'][M1[condition1,0]].T,sample_2['points'][M1[condition1,1]].T).T
        # # print(points3d_1[:,-1].shape)
        # points3d_1 = (points3d_1 /  points3d_1[:,-1].reshape(-1,1))[:,:-1]

        # M2 = nn_match_two_way(sample_2['desc'].T, sample_3['desc'].T,0.2)
        # M2 = M2.astype(np.int32)
        # e2,mask2=cv2.findEssentialMat(sample_2['points'][M2[:,0]],sample_3['points'][M2[:,1]],camera_mat.astype(np.double),method=cv2.RANSAC, prob=0.999, threshold=0.001)
        # condition2 = mask2 == 1
        # condition2 = condition2.reshape((M2.shape[0],))
        
        # _,R2,t2,_ = cv2.recoverPose(e2,sample_2['points'][M2[condition2,0]],sample_3['points'][M2[condition2,1]])
        # trans2[:-1,:-1] = R2
        # trans2[:-1,-1] = t2.reshape((3,))
        # points3d_2 = cv2.triangulatePoints(trans[:-1,:],trans2[:-1,:],sample_2['points'][M2[condition2,0]].T,sample_3['points'][M2[condition2,1]].T).T
        # points3d_2 = (points3d_2 /  points3d_2[:,-1].reshape(-1,1))[:,:-1]



        # r = np.linalg.norm(points3d_1[:10] - points3d_1[10:20]) / np.linalg.norm(points3d_2[:10] - points3d_2[10:20])

        # print(r)
    


    pass    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Odometry using superpoint matching''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/apples_big_2021-10-14-14-51-08_0/', type=str,
                        help='Path to the root directory of the dataset')

    # parser.add_argument('-v', '--visualize', default=False,
    #                     type=bool, help='Visualize results')
    # parser.add_argument('-s', '--save', default=False,
    #                     type=bool, help='Save flag')
    args = parser.parse_args()
    dataset = DatasetOdometry(args.data_root_path)
   
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_mat = np.asarray(rgb_camera_intrinsic.intrinsic_matrix)
    if "right" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    
    I = np.eye(4)

   

    trans_init = np.eye(4)
    sample_1 = dataset[0]
    sample_2 = dataset[1]


    obtain_relative_scale(dataset,camera_mat)


    
    # M = nn_match_two_way(sample_1['desc'].T, sample_2['desc'].T,0.4)
    # M = M.astype(np.int32)


    # e,mask=cv2.findEssentialMat(sample_1['points'][M[:,0]],sample_2['points'][M[:,1]],camera_mat.astype(np.double),method=cv2.RANSAC, prob=0.999, threshold=0.05)
    # condition = mask == 1
    # condition = condition.reshape((M.shape[0],))
    # distortion_mat = np.array([-0.057091124355793,0.06687477231025696,-8.671214891364798e-05,-6.639014463871717e-05,-0.021493962034583092])
    
    
    # _,R,t,_ = cv2.recoverPose(e,sample_1['points'][M[condition,0]],sample_2['points'][M[condition,1]])
    # trans_init[:-1,:-1] = R
    # trans_init[:-1,-1] = t.reshape((3,))

    # print(f"Trans init:\n{trans_init}\n")



    # rgb1_o3d = o3d.geometry.Image(sample_1['rgb'])
    # depth1_o3d = o3d.geometry.Image(sample_1['depth'])
    # rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         rgb1_o3d,
    #         depth1_o3d,
    #         depth_scale=1000,
    #         depth_trunc=5.0,
    #         convert_rgb_to_intensity=False,
    #     )
    
    # rgb2_o3d = o3d.geometry.Image(sample_2['rgb'])
    # depth2_o3d = o3d.geometry.Image(sample_2['depth'])
    # rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         rgb2_o3d,
    #         depth2_o3d,
    #         depth_scale=1000,
    #         depth_trunc=5.0,
    #         convert_rgb_to_intensity=False,
    #     )

    # target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #         rgbd1, rgb_camera_intrinsic, I)
    # target_pcd.estimate_normals()
    
    # source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #         rgbd2, rgb_camera_intrinsic, I
    #     )
    
    # result = o3d.pipelines.registration.registration_icp(
    #         source_pcd,
    #         target_pcd,
    #         0.05,
    #         np.eye(4),
    #         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     )
    
    # print(f"Result ICP:\n{result.transformation}\n")

    # print(f"Norm : {np.linalg.norm(result.transformation[:-1,-1]-trans_init[:-1,-1])}")
    # o3d.visualization.draw_geometries([target_pcd, source_pcd.transform(trans_init)])








    






