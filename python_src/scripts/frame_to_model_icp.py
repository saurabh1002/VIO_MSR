#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import os
import argparse
from pyexpat import model
from tqdm import tqdm
import sys; sys.path.append(os.pardir)

from utils.utils import *
from superpoint_matching import nn_match_two_way
from utils.dataloader import DatasetOdometry, DatasetOdometryAll

import ipdb
import open3d as o3d
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.spatial.transform as tf

def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:-1, :-1]).as_quat()
    return list(np.r_[t, origin, rot_quat])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Point-to-Point ICP Odometry''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize output')
    parser.add_argument('-d', '--debug', default=False, type=bool, help='Debug Flag')
    parser.add_argument('-p', '--plot', default=True, type=bool, help='Plot the odometry results')

    args = parser.parse_args()
    skip = args.skip_frames

    dataset_name = 'apples_big_2021-10-14-all/'

    if not "all" in dataset_name:
        dataset = DatasetOdometry(args.data_root_path + dataset_name)
    else:
        dataset = DatasetOdometryAll(args.data_root_path + dataset_name)

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()

    if "right" in args.data_root_path:
        cam_intrinsics.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)

    K = cam_intrinsics.intrinsic_matrix

    T = np.eye(4)
    poses = []
    poses.append(odom_from_SE3(dataset[0]['timestamp'], T))

    I = np.eye(4)
    T = np.eye(4)
    T_model = np.eye(4)
    poses = []
    n = 0

    models = []
    model_poses = []

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    t, rgb_cv2, depth_cv2 = dataset[0]['timestamp'], dataset[0]['rgb'], dataset[0]['depth']
    rgb_o3d = o3d.geometry.Image(rgb_cv2)
    depth_o3d = o3d.geometry.Image(depth_cv2)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1000,
        depth_trunc=5.0,
        convert_rgb_to_intensity = False
    )
    volume.integrate(rgbd, cam_intrinsics, I)

    progress_bar = tqdm(range(skip, len(dataset) - skip, skip))
    for i in progress_bar:
        t = dataset[i]['timestamp']
        rgb_o3d = o3d.geometry.Image(dataset[i]['rgb'])
        depth_o3d = o3d.geometry.Image(dataset[i]['depth'])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1000,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )

        model_pcd = volume.extract_point_cloud()

        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, cam_intrinsics, I
        )

        result = o3d.pipelines.registration.registration_icp(
            frame_pcd,
            model_pcd,
            0.05,
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        T = result.transformation

        volume.integrate(
            rgbd,
            cam_intrinsics,
            la.inv(T),
        )
        poses.append(odom_from_SE3(t, np.array(T)))
    
        if i % 250 == 0:
            o3d.io.write_triangle_mesh(f"../../eval_data/front/{dataset_name}mesh_model_to_frame_{n}.ply", volume.extract_triangle_mesh())
            model_poses.append(T_model)
            n = n + 1

            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.01,
                sdf_trunc=0.05,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                )

            t, rgb_cv2, depth_cv2 = dataset[i]['timestamp'], dataset[i]['rgb'], dataset[i]['depth']
            rgb_o3d = o3d.geometry.Image(rgb_cv2)
            depth_o3d = o3d.geometry.Image(depth_cv2)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale=1000,
                depth_trunc=5.0,
                convert_rgb_to_intensity = False
            )
            volume.integrate(rgbd, cam_intrinsics, la.inv(T))
            T_model = T

    poses = np.asarray(poses)
    np.savetxt(f"../../eval_data/front/{dataset_name}poses_model_to_frame_skip{skip}.txt", poses)
    np.savetxt(f"../../eval_data/front/{dataset_name}poses_model{skip}.txt", model_poses)

    if args.plot:
        x = poses[:, 1]
        y = poses[:, 2]
        z = poses[:, 3]

        gt = np.loadtxt(args.data_root_path + "groundtruth.txt")
        gt_x = gt[:, 1]
        gt_y = gt[:, 2]
        gt_z = gt[:, 3]

        fig,(ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(-x, y, 'r', label='Estimated')
        ax1.plot(gt_y, gt_z ,'b', label='Ground Truth')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('Z')
        ax1.legend()
        ax2.plot(z, y, 'r', label='Estimated')
        ax2.plot(gt_x, gt_z, 'b', label='Ground Truth')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.legend()
        ax3.plot(z, -x, 'r', label='Estimated')
        ax3.plot(gt_x, gt_y, 'b', label='Ground Truth')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"../../eval_data/front/{dataset_name}poses_ICP_Full_skip{skip}.png")
        plt.show()

    # for n in range(len(models)):
    #     o3d.io.write_triangle_mesh("../../eval_data/front/{dataset_name}mesh_model_to_frame_{n}.ply", models[n])

    # if args.visualize:
    #     o3d.visualization.draw_geometries(models)