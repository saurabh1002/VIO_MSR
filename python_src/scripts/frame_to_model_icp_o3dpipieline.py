#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
import scipy.spatial.transform as tf

from tqdm import tqdm

import sys
import time

sys.path.append("..")
from utils.dataloader import DatasetRGBD

import open3d as o3d


def slam(timestamps, depth_file_names, color_file_names, intrinsic):
    n_files = len(color_file_names)
    device = o3d.core.Device("CUDA:0")

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(0.01, 16,
                                       40000, T_frame_to_model,
                                       device)
    depth_ref = o3d.t.io.read_image(depth_file_names[0])
    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                             intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                               depth_ref.columns, intrinsic,
                                               device)

    poses = []

    for i in range(n_files):
        start = time.time()

        t = timestamps[i]
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)

        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)

        if i > 0:
            result = model.track_frame_to_model(input_frame, raycast_frame,
                                                1000,
                                                5.0)
            T_frame_to_model = T_frame_to_model @ result.transformation

        poses.append(odom_from_SE3(t, T_frame_to_model.cpu().numpy()))
        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(input_frame, 1000, 5.0)
        model.synthesize_model_frame(raycast_frame, 1000,
                                     0.1, 5.0, False)
        stop = time.time()
        print('{:04d}/{:04d} slam takes {:.4}s'.format(i, n_files,
                                                       stop - start))

    return model.voxel_grid, poses

def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:3, :3]).as_quat()
    return list(np.r_[t, origin, rot_quat])


if __name__ == "__main__":

    datadir = "../../datasets/phenorob/front/apples_big_2021-10-14-all/"

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if "right" in datadir:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in datadir:
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    
    data = DatasetRGBD(datadir)

    volume, poses = slam(data.timestamps_rgb[:5000:5], data.depth_paths[:5000:5], data.rgb_paths[:5000:5], o3d.core.Tensor(rgb_camera_intrinsic.intrinsic_matrix))

    np.savetxt("../../eval_data/front/apples_big_2021-10-14-all/poses_model_to_frame.txt", np.array(poses))
    mesh = volume.extract_triangle_mesh()
    mesh = mesh.to_legacy()
    # mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("../../eval_data/front/apples_big_2021-10-14-all/full_mesh_model_to_frame.ply", mesh)
    o3d.visualization.draw_geometries([mesh])