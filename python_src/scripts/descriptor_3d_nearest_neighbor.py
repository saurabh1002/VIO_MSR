import os
import sys
sys.path.append(os.pardir)
import pickle
import argparse
from typing import Tuple

import cv2
import open3d as o3d
import scipy.spatial
import numpy as np
import numpy.linalg as la

from utils.ransac_homography import *
from utils.utils import wrapTo2Pi


def process_input_data(bboxes_path: str, associations_path: str) -> Tuple[dict, list, list]:
    ''' Loads the input data for further use

    Arguments
    ---------
    - bboxes_path: Path to the pickle file containing the bounding box data for each rgb frame
    - associations_path: Path to the txt file conatining the associations for RGB and Depth frames from the dataset
    
    Returns
    -------
    - bboxes_data: {rgb_frame_path: bboxes} a dictionary containing the bounding boxes in each rgb frame as key-value pairs
    - rgb_frame_names: list containing filenames for all rgb frames from the associations file
    - depth_frame_names: list containing filenames for all depth frames from the associations file
    '''
    with open(bboxes_path, "rb") as f:
        bboxes_data = pickle.load(f)
    
    with open(associations_path, 'r') as f:
        depth_frame_names = []
        rgb_frame_names = []

        for line in f.readlines():
            _, rgb_path, _, depth_path = line.rstrip("\n").split(' ')
            if(os.path.basename(rgb_path) in list(bboxes_data.keys())):
                depth_frame_names.append(depth_path)
                rgb_frame_names.append(rgb_path)

    return bboxes_data, rgb_frame_names, depth_frame_names

def get_depth_mask(depth_frame_shape: tuple, keypoint: np.ndarray) -> (np.ndarray):
    '''This function creates a depth mask for a keypoint pixel

    Arguments
    ---------
    - depth_frame_shape: shape of the depth frame
    - keypoint: keypoint of interest

    Returns
    -------
    - depth_mask: depth mask for a pixel
    '''
    depth_mask = np.zeros(depth_frame_shape, np.uint8)
    depth_mask[int(keypoint[1]), int(keypoint[0])] = 1
    return depth_mask

def get_keypoints(depth_frame: np.ndarray, keypts_2d: np.ndarray, rgb_camera_intrinsic: o3d._pybind_cuda,
    depth_scale: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
    ''' Computes 3D keypoints from the detection centers in the RGB frame and associated depth frame
    
    Arguments
    ---------
    - depth_frame: Depth frame to be processed
    - keypts_2d: 2D keypoints in the frame
    - rgb_camera_intrinsic: Intrinsic parameters of the RGB camera in open3d format
    - depth_scale: scale factor for the depth camera [m <-> pixel value]

    Returns
    -------
    - keypts_2d: 2D keypoints in the frame
    - keypts_3d: corresponding 3D keypoints in the frame

    '''
    keypts_3d = []
    pcl_gen_flag = []

    for keypt in keypts_2d:
        depth_mask = get_depth_mask(depth_frame.shape, keypt)
        depth_image = o3d.geometry.Image(np.multiply(depth_frame, depth_mask))
        pcl = o3d.geometry.PointCloud.create_from_depth_image(depth_image, rgb_camera_intrinsic, depth_scale=depth_scale)

        # Check if 2d keypoint could be succesfully projected to 3d
        pcl_gen_flag.append(pcl.has_points())
        if(pcl.has_points()):
            keypts_3d.append(np.array(pcl.points)[0])
    
    keypts_2d = np.array(keypts_2d)[pcl_gen_flag]
    keypts_3d = np.array(keypts_3d)

    return keypts_2d, keypts_3d

def find_k_nearest(keypoints_3d: np.ndarray, k: int) -> (np.ndarray):
    ''' Compute the k nearest keypoints to the current keypoint of interest
    '''
    if keypoints_3d.shape[0] > (k + 1):
        dist = scipy.spatial.distance.cdist(keypoints_3d, keypoints_3d)
        sort_ids = np.argsort(dist)
        return sort_ids[:, 1:k+1]
    else:
        raise RuntimeError("Not enough features to compute descriptor")
    
def compute_descriptor(keypoints_2d: np.ndarray, keypoints_3d: np.ndarray, knn_ids: np.ndarray) -> (np.ndarray):
    ''' Compute Descriptor for each 3d feature in the RGB-D frame

    Arguments
    ---------
    - keypoints_2d: numpy array of all 2D keypoints in the RGB frame
    - keypoints_3d: numpy array of all 3D keypoints in the RGB-D frame
    - knn_ids: indices corresponding to the k nearest neigbors of each 3D keypoint

    Returns
    -------
    - descriptor: [2D location, distance ratios, angles in 3D]
    '''
    k = knn_ids.shape[1]
    descriptors = np.zeros((keypoints_3d.shape[0], k * 2))
    for i, kp in enumerate(keypoints_3d):
        dist_3d = la.norm(kp - keypoints_3d[knn_ids[i]], 2, 1)
        dist_ratios_3d = dist_3d[:-1] / dist_3d[-1]
        
        angles_2d = np.arctan2(keypoints_2d[knn_ids[i, :-1], 1] - keypoints_2d[i, 1], 
            keypoints_2d[knn_ids[i, :-1], 0] - keypoints_2d[i, 0]) - np.arctan2(keypoints_2d[knn_ids[i, -1], 1] - keypoints_2d[i, 1], 
            keypoints_2d[knn_ids[i, -1], 0] - keypoints_2d[i, 0]) 
        angles_2d = np.array([wrapTo2Pi(-angle) for angle in angles_2d]) / (2 * np.pi)

        sort_idx = np.argsort(angles_2d)

        dot_product = (keypoints_3d[knn_ids[i, :-1]] - kp) @ (keypoints_3d[knn_ids[i, -1]] - kp)
        mag = la.norm(keypoints_3d[knn_ids[i, :-1]] - kp, 2, 1) * la.norm(keypoints_3d[knn_ids[i, -1]] - kp, 2)
        angles_3d = np.arccos(np.divide(dot_product, mag)) / np.pi

        descriptors[i] = np.r_[keypoints_2d[i], dist_ratios_3d[sort_idx], angles_3d[sort_idx]]
    return descriptors


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''
    This script generates custom descriptors for 3D macro keypoints
    ''')
    # Dataset paths
    parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/bboxes.pickle', type=str,
        help='Path to the centernet object detection bounding box coordinates')
    parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations_rgbd.txt', type=str,
        help='Path to the associations file for RGB and Depth frames')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
        help='Path to the root directory of the dataset')

    parser.add_argument('--k', default=4, type=int, help='Number of nearest neighbors to use for descriptor')
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save_descriptor', default=False, type=bool, help='Save computed Descriptors as .npz file')
    args = parser.parse_args()

    bboxes_d, rgb_names, depth_names = process_input_data(args.bboxes_path, args.associations_path)
     
    num_of_frames = len(rgb_names)
    
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if args.data_root_path == "../../datasets/phenorob/images_apples_right/":
        rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif args.data_root_path == '../../datasets/phenorob/images_apples/':
        rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    
    for n in range(num_of_frames - 1):
        rgb_frame_1 = cv2.imread(args.data_root_path + rgb_names[n])
        depth_frame_1 = cv2.imread(args.data_root_path + depth_names[n], cv2.CV_16UC1)
        keypts_2d_1 = np.array(bboxes_d[os.path.basename(rgb_names[n])])
        keypts_2d_1, keypts_3d_1 = get_keypoints(depth_frame_1, keypts_2d_1, rgb_camera_intrinsic)

        rgb_frame_2 = cv2.imread(args.data_root_path + rgb_names[n + 1])
        depth_frame_2 = cv2.imread(args.data_root_path + depth_names[n + 1], cv2.CV_16UC1)
        keypts_2d_2 = np.array(bboxes_d[os.path.basename(rgb_names[n + 1])])
        keypts_2d_2, keypts_3d_2 = get_keypoints(depth_frame_2, keypts_2d_2, rgb_camera_intrinsic)

        knn_id_1 = find_k_nearest(keypts_3d_1, args.k)
        knn_id_2 = find_k_nearest(keypts_3d_2, args.k)

        descriptors_1 = compute_descriptor(keypts_2d_1, keypts_3d_1, knn_id_1)
        descriptors_2 = compute_descriptor(keypts_2d_2, keypts_3d_2, knn_id_2)

        M = compute_matches(descriptors_1[:, 2:], descriptors_2[:, 2:], rgb_frame_1.shape[0], rgb_frame_1.shape[1])

        P1 = descriptors_1[:, :2].astype(int)
        P2 = descriptors_2[:, :2].astype(int)

        if(M.shape[0] > 0):
            H, M = compute_homography_ransac(P1, P2, M)

        w = rgb_frame_1.shape[1]
        rgb_match_frame = np.concatenate((rgb_frame_1, rgb_frame_2), 1)

        for kp in descriptors_1:
            cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 10, (255, 0, 0), 2)

        for kp in descriptors_2:
            cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                       int(kp[1])), 10, (0, 0, 255), 2)

        for kp_l, kp_r in zip(descriptors_1[M[:, 0]].astype(int), descriptors_2[M[:, 1]].astype(int)):
            cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)

        if args.visualize:
            cv2.imshow('matches', rgb_match_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.save_descriptor:
            cv2.imwrite(
                '../../eval_data/custom_3d_desc/{}.png'.format(n), rgb_match_frame)