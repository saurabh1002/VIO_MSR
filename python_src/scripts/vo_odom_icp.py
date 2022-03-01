import os
import argparse
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

def getMatches(keypts_1: dict, keypts_2: dict, threshold: float = 0.7):
    
    points2d_1 = keypts_1['points'][:,:2]
    points2d_2 = keypts_2['points'][:,:2]

    descriptor_1 = keypts_1['desc']
    descriptor_2 = keypts_2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T, threshold).astype(np.int64)

    return points2d_1[M[:, 0]], points2d_2[M[:, 1]]

def convertTo3d(depth_frame: np.ndarray, keypoints_2d: np.ndarray, K: np.ndarray):

    depth_factor = 1000
    permute_col_2d = np.array([[0, 1],[1, 0]])
    permuted_keypoints = keypoints_2d @ permute_col_2d
    keypts_depth = depth_frame[permuted_keypoints[:, 0], permuted_keypoints[:, 1]] / depth_factor

    transformed_points = toHomogenous(keypoints_2d)
    ray_dir = transformed_points @ la.inv(K).T
    keypoints_3d = np.multiply(ray_dir, keypts_depth.reshape(-1, 1))
    
    return keypoints_3d


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

    device = o3d.core.Device("CUDA:0")
    tensor = True

    dataset_name = 'apples_big_2021-10-14-14-51-08_0/'
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

    p2p_distance_threshold = 0.05
    initial_estimate_tf = np.eye(4)
    
    min_dist = 0.01
    skip = args.skip_frames
    i = 0
    j = i + skip

    while True:
        try:
            keypts_2d_1, keypts_2d_2 = getMatches(dataset[i], dataset[j])    
            
            keypts_3d_1 = convertTo3d(dataset[i]['depth'], keypts_2d_1.astype(np.int64), K)
            keypts_3d_2 = convertTo3d(dataset[j]['depth'], keypts_2d_2.astype(np.int64), K)
            
            rgbd1 = create_rgbdimg(dataset[i]['rgb'], dataset[i]['depth'], tensor=tensor)
            rgbd2 = create_rgbdimg(dataset[j]['rgb'], dataset[j]['depth'], tensor=tensor)

            target_pcd = RGBD2PCL(rgbd1, cam_intrinsics, compute_normals=True, tensor=tensor)
            source_pcd = RGBD2PCL(rgbd2, cam_intrinsics, compute_normals=False, tensor=tensor)

            # target_pcd = o3d.t.geometry.PointCloud(device)
            # target_pcd.point["positions"] = o3d.core.Tensor(keypts_3d_1, o3d.core.Dtype.Float64, device)

            # source_pcd = o3d.t.geometry.PointCloud(device)
            # source_pcd.point["positions"] = o3d.core.Tensor(keypts_3d_2, o3d.core.Dtype.Float64, device)

            result = o3d.t.pipelines.registration.icp(
                source_pcd,
                target_pcd,
                p2p_distance_threshold,
                o3d.core.Tensor(initial_estimate_tf, o3d.core.Dtype.Float64),
                o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            if la.norm(result.transformation.numpy()[:3, -1]) < min_dist:
                j = j + 1
                continue

            T = result.transformation.numpy() @ T
            i = j
            j = j + skip

            # o3d.visualization.draw_geometries([target_pcd.to_legacy().paint_uniform_color(np.array([1, 0, 0])),
            #                                    source_pcd.to_legacy().paint_uniform_color(np.array([0, 1, 0]))])

            # o3d.visualization.draw_geometries([target_pcd.to_legacy().paint_uniform_color(np.array([1, 0, 0])),
            #                                    source_pcd.transform(result.transformation).to_legacy().paint_uniform_color(np.array([0, 1, 0]))])

            poses.append(odom_from_SE3(dataset[i]['timestamp'], T))

            if args.debug:
                pass
        except IndexError:
            break

    poses = np.asarray(poses)
    np.savetxt(f"../../eval_data/front/{dataset_name}poses_ICP_Full_skip{skip}.txt", poses)

    # ipdb.set_trace()

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

    # np.savetxt(args.data_root_path + "pose.txt", np.array(poses))
    
    # T_ = icp_known_corresp(keypts_3d_1.T,keypts_3d_2.T,idx,idx)

    # # print(type(points_3d))
    # # import ipdb;ipdb.set_trace()
    # rgbd1 = create_rgbdimg(dataset[i]['rgb'], dataset[i]['depth'])
    # source_pcd = RGBD2PCL(rgbd1,rgb_camera_intrinsic,False)
    # pcl1 = o3d.geometry.PointCloud()
    # pcl1.points = o3d.utility.Vector3dVector(keypts_3d_1)
    # pcl1.paint_uniform_color([0, 0, 1])
    # pcl2 = o3d.geometry.PointCloud()
    # pcl2.points = o3d.utility.Vector3dVector(keypts_3d_2)
    # pcl2.paint_uniform_color([1, 0, 0])
    
    # result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(pcl2,pcl1,o3d.utility.Vector2iVector(np.dstack((idx,idx))[0]))
    
    # T = T.dot(result.transformation)