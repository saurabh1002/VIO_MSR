#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

from typing import Tuple
import numpy as np
import open3d as o3d
import pickle
import cv2


class DatasetPCL:
    def __init__(self, datadir):
        with open(datadir + "associations_pcl_gt.txt", "r") as f:
            self.timestamps_pcl = []
            self.scans_path = []
            self.origins = []
            self.rot_quat = []
            for line in f.readlines():
                t_pcl, pcl_path, _, tx, ty, tz, qx, qy, qz, qw = line.rstrip(
                    "\n"
                ).split(" ")
                self.timestamps_pcl.append(float(t_pcl))
                self.origins.append([float(tx), float(ty), float(tz)])
                self.rot_quat.append(
                    [float(qw), float(qx), float(qy), float(qz)])
                self.scans_path.append(datadir + pcl_path)

    def __len__(self) -> int:
        return len(self.scans_path)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        timestamp = self.timestamps_pcl[idx]
        points = o3d.io.read_point_cloud(self.scans_path[idx])
        origin = np.array(self.origins[idx])
        rot = np.array(self.rot_quat[idx])
        return timestamp, points, origin, rot


class DatasetRGBD:
    def __init__(self, datadir):
        with open(datadir + "associations_rgbd.txt", "r") as f:
            self.rgb_paths = []
            self.depth_paths = []
            self.timestamps_rgb = []
            self.timestamps_depth = []
            for line in f.readlines():
                t_rgb, rgb_file, t_depth, depth_file = line.rstrip(
                    "\n").split(" ")
                self.timestamps_rgb.append(float(t_rgb))
                self.rgb_paths.append(datadir + rgb_file)
                self.timestamps_depth.append(float(t_depth))
                self.depth_paths.append(datadir + depth_file)

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        timestamp = self.timestamps_rgb[idx]
        rgb_frame = cv2.imread(self.rgb_paths[idx])
        depth_frame = cv2.imread(self.depth_paths[idx], cv2.CV_16UC1)
        return timestamp, rgb_frame, depth_frame


class DatasetGroundTruth:
    def __init__(self, datadir):
        with open(datadir + "associations_rgbd_gt.txt", "r") as f:
            self.rgb_paths = []
            self.depth_paths = []
            self.timestamps_rgb = []
            self.origins = []
            self.rot_quat = []
            for line in f.readlines():
                t_rgb, rgb_file, _, depth_file, _, tx, ty, tz, qx, qy, qz, qw = line.rstrip("\n").split(
                    " "
                )
                self.timestamps_rgb.append(float(t_rgb))
                self.origins.append([float(tx), float(ty), float(tz)])
                self.rot_quat.append(
                    [float(qw), float(qx), float(qy), float(qz)])
                self.rgb_paths.append(datadir + rgb_file)
                self.depth_paths.append(datadir + depth_file)

    def __len__(self) -> int:
        return len(self.timestamps_rgb)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        timestamp = self.timestamps_rgb[idx]
        rgb_frame = cv2.imread(self.rgb_paths[idx])
        depth_frame = cv2.imread(self.depth_paths[idx], cv2.CV_16UC1)
        origin = np.array(self.origins[idx])
        rot = np.array(self.rot_quat[idx])
        return timestamp, rgb_frame, depth_frame, origin, rot


class DatasetOdometry:

    def __init__(self, datadir):
        self.data_root = datadir
        self.rgb_frame_names = []
        self.depth_frame_names = []
        self.timestamps = []
        with open(self.data_root + 'associations_rgbd.txt', 'r') as f:
            for line in f.readlines():
                timestamp, rgb_path, _, depth_path = line.rstrip("\n").split(' ')
                self.rgb_frame_names.append(rgb_path)
                self.depth_frame_names.append(depth_path)
                self.timestamps.append(timestamp)
        with open(self.data_root + 'superpoint/' + 'points.pickle', "rb") as f:
            self.points_all = pickle.load(f)
        with open(self.data_root + 'superpoint/' + 'descriptors.pickle', "rb") as f:
            self.descriptors_all = pickle.load(f)

    def __len__(self):
        return len(self.rgb_frame_names)

    def __getitem__(self, idx):

        rgb_frame = cv2.imread(self.data_root + self.rgb_frame_names[idx])
        depth_frame = cv2.imread(self.data_root + self.depth_frame_names[idx],cv2.CV_16UC1)
        points = self.points_all[self.rgb_frame_names[idx].split(
            '/')[1]]
        descriptor = self.descriptors_all[self.rgb_frame_names[idx].split('/')[1]]
        timestamp = self.timestamps[idx]
        sample = {'rgb': rgb_frame, 'depth': depth_frame,
                  'points': points, 'desc': descriptor,'timestamp':timestamp}

        return sample
