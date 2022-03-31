import os
import sys
sys.path.append(os.pardir)
import argparse
from tqdm import tqdm

from utils.utils import *
from utils.dataloader import DatasetOdometry, DatasetOdometryAll

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def argparser():
    """Argument Parser
    """
    parser = argparse.ArgumentParser(description='''Pose Estimation using Frame-to-Frame ICP''')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-c', '--cuda', default=False, type=bool, help="Use Tensor mode for Open3D with Cuda")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize output')
    parser.add_argument('-d', '--debug', default=False, type=bool, help='Debug Flag')
    parser.add_argument('-p', '--plot', default=True, type=bool, help='Plot the odometry results')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = argparser()
    dataset_name = 'apples_big_2021-10-14-all/'

    tensor = args.cuda

    if not "all" in dataset_name:
        dataset = DatasetOdometry(args.data_root_path + dataset_name)
    else:
        dataset = DatasetOdometryAll(args.data_root_path + dataset_name)

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()

    if "right" in args.data_root_path:
        cam_intrinsics.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)

    T_rot =  (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @
                np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])).T

    T = np.eye(4)
    poses = []
    poses.append(odom_from_SE3(dataset[0]['timestamp'], T))

    p2p_distance_threshold = 0.05
    initial_estimate_tf = np.eye(4)
    
    skip = args.skip_frames

    i = 0
    j = i + skip

    pbar = tqdm(total = len(dataset) - 1)
    while (j < len(dataset)):
        rgbd1 = create_rgbd_from_color_and_depth(cv2.cvtColor(dataset[i]['rgb'], cv2.COLOR_BGR2RGB), 
            dataset[i]['depth'], tensor=tensor)
        rgbd2 = create_rgbd_from_color_and_depth(cv2.cvtColor(dataset[j]['rgb'], cv2.COLOR_BGR2RGB),
            dataset[j]['depth'], tensor=tensor)

        target_pcd = compute_pcl_from_rgbd(rgbd1, cam_intrinsics, 
            compute_normals=False, tensor=tensor)
        source_pcd = compute_pcl_from_rgbd(rgbd2, cam_intrinsics, 
            compute_normals=False, tensor=tensor)

        if tensor:
            result = o3d.t.pipelines.registration.icp(
                source_pcd,
                target_pcd,
                p2p_distance_threshold,
                o3d.core.Tensor(initial_estimate_tf, o3d.core.Dtype.Float64),
                o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
            )

        else:
            result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                p2p_distance_threshold,
                initial_estimate_tf,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

        T = T @ result.transformation.numpy()

        i = j
        j = j + skip
        pbar.update(skip)

        if args.visualize:
            if tensor:
                o3d.visualization.draw_geometries([target_pcd.to_legacy().paint_uniform_color(np.array([1, 0, 0])),
                                                source_pcd.transform(result.transformation).to_legacy().paint_uniform_color(np.array([0, 1, 0]))])
            else:
                o3d.visualization.draw_geometries([target_pcd.paint_uniform_color(np.array([1, 0, 0])),
                                                source_pcd.transform(result.transformation).paint_uniform_color(np.array([0, 1, 0]))])
                
        poses.append(odom_from_SE3(dataset[i]['timestamp'], T, T_rot))

    poses = np.asarray(poses)

    cam_dir = 'front' if 'front' in args.data_root_path else 'right'

    if args.plot:
        x = poses[:, 1]
        y = poses[:, 2]
        z = poses[:, 3]

        gt = np.loadtxt(args.data_root_path + "groundtruth.txt")
        gt_x = gt[:, 1]
        gt_y = gt[:, 2]
        gt_z = gt[:, 3]

        fig,(ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(y, z, 'r', label='frame-to-frame ICP')
        ax1.plot(gt_y, gt_z ,'b', label='liosam')
        ax1.set_xlabel('Y')
        ax1.set_ylabel('Z')
        ax1.legend()
        ax2.plot(x, z, 'r', label='frame-to-frame ICP')
        ax2.plot(gt_x, gt_z, 'b', label='liosam')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.legend()
        ax3.plot(x, y, 'r', label='frame-to-frame ICP')
        ax3.plot(gt_x, gt_y, 'b', label='liosam')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"../../eval_data/{cam_dir}/{dataset_name}ICP/poses_frame_to_frame_skip_{skip}.png")
        plt.show()

    np.savetxt(f"../../eval_data/{cam_dir}/{dataset_name}ICP/poses_frame_to_frame_skip_{skip}.txt", poses)