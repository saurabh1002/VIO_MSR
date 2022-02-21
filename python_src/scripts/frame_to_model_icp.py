#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
import numpy.linalg as la
import scipy.spatial.transform as tf

import argparse
from tqdm import tqdm
import sys; sys.path.append("..")

from utils.dataloader import DatasetRGBD

import open3d as o3d

def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:3, :3]).as_quat()
    return list(np.r_[t, origin, rot_quat])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Odometry using superpoint matching''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize output')
    parser.add_argument('-d', '--debug', default=False, type=bool, help='Debug Flag')
    parser.add_argument('-p', '--plot', default=True, type=bool, help='Plot the odometry results')

    args = parser.parse_args()

    dataset_name = 'apples_big_2021-10-14-all/'

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()

    if "right" in args.data_root_path:
        cam_intrinsics.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    
    data = DatasetRGBD(args.data_root_path + dataset_name)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    I = np.eye(4)
    T = np.eye(4)
    poses = []

    t, rgb_cv2, depth_cv2 = data[0]
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

    for i, [t, rgb_cv2, depth_cv2] in tqdm(enumerate(data)):
        rgb_o3d = o3d.geometry.Image(rgb_cv2)
        depth_o3d = o3d.geometry.Image(depth_cv2)
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

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    np.savetxt(f"../../eval_data/front/{dataset_name}poses_model_to_frame.txt", np.array(poses))
    o3d.io.write_triangle_mesh("../../eval_data/front/{dataset_name}mesh_model_to_frame.ply", mesh)
    if args.visualize:
        o3d.visualization.draw_geometries([mesh])