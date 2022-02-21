import os
import argparse
from tqdm import tqdm
import sys; sys.path.append(os.pardir)

from utils.dataloader import DatasetOdometry, DatasetOdometryAll
from utils.utils import *

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

    skip = args.skip_frames
    progress_bar = tqdm(range(0, len(dataset) - skip, skip))
    for i in progress_bar:
        rgbd1 = create_rgbdimg(dataset[i]['rgb'], dataset[i]['depth'])
        rgbd2 = create_rgbdimg(dataset[i + skip]['rgb'], dataset[i + skip]['depth'])

        source_pcd = RGBD2PCL(rgbd1, cam_intrinsics, compute_normals=False)
        target_pcd = RGBD2PCL(rgbd2, cam_intrinsics, compute_normals=False)

        p2p_distance_threshold = 0.05
        initial_estimate_tf = np.eye(4)
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            p2p_distance_threshold,
            initial_estimate_tf,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        T = T @ result.transformation

        poses.append(odom_from_SE3(dataset[i]['timestamp'], T))

        if args.debug:
            pass

    poses = np.asarray(poses)
    np.savetxt(f"../../eval_data/front/{dataset_name}poses_ICP_Full_skip{skip}.txt", poses)

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
        ax1.legend()
        ax2.plot(z, y, 'r', label='Estimated')
        ax2.plot(gt_x, gt_z, 'b', label='Ground Truth')
        ax2.legend()
        ax3.plot(z, -x, 'r', label='Estimated')
        ax3.plot(gt_x, gt_y, 'b', label='Ground Truth')
        ax3.legend()

        plt.show()

    # np.savetxt(args.data_root_path + "pose.txt", np.array(poses))
    
    # T_ = icp_known_corresp(keypts_3d_1.T,keypts_3d_2.T,idx,idx)

    # import ipdb;ipdb.set_trace()
    # # print(type(points_3d))
    # # import ipdb;ipdb.set_trace()
    # rgbd1 = create_rgbdimg(dataset[i]['rgb'],dataset[i]['depth'])
    # source_pcd = RGBD2PCL(rgbd1,rgb_camera_intrinsic,False)
    # pcl1 = o3d.geometry.PointCloud()
    # pcl1.points = o3d.utility.Vector3dVector(keypts_3d_1)
    # pcl1.paint_uniform_color([0, 0, 1])
    # pcl2 = o3d.geometry.PointCloud()
    # pcl2.points = o3d.utility.Vector3dVector(keypts_3d_2)
    # pcl2.paint_uniform_color([1, 0, 0])
    
    # result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(pcl2,pcl1,o3d.utility.Vector2iVector(np.dstack((idx,idx))[0]))
    
    # T = T.dot(result.transformation)