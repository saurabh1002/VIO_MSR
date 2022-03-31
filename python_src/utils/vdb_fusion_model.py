import os
import pickle
import argparse
from tqdm import tqdm
from typing import Tuple

import numpy as np
import open3d as o3d
from vdbfusion import VDBVolume

from utils import *

class DatasetOdometryAll:
    def __init__(self, datadir: str, type: str = 'superpoint'):
        self.data_root = datadir
        self.type = type
        self.rgb_frame_names = []
        self.depth_frame_names = []
        self.timestamps = []
        self.folder_names = []
        
        with open(self.data_root + 'associations_rgbd.txt', 'r') as f:
            for line in f.readlines():
                timestamp, rgb_path, _, depth_path = line.rstrip("\n").split(' ')
                self.rgb_frame_names.append(rgb_path[3:])
                self.depth_frame_names.append(depth_path[3:])
                self.timestamps.append(float(timestamp))
                name = rgb_path.split("/")[1]

                if name not in self.folder_names:
                    self.folder_names.append(name)
                else:
                    continue
            
            self.root_name = os.path.dirname(os.path.abspath(self.data_root))
            
            if self.type == 'superpoint':
                self.points_all = {}
                self.descriptors_all = {}
                for name in self.folder_names:
                    superpoint_path = os.path.join(self.root_name, name, "superpoint/")
                    points, descriptors = process_superpoint_feature_descriptors(superpoint_path)
                    self.points_all = {**self.points_all, **points}
                    self.descriptors_all = {**self.descriptors_all, **descriptors}
            
            elif self.type == '3Dhist':
                self.detections_all = {}
                for name in self.folder_names:
                    detections_path = os.path.join(self.root_name, name, "detections/")
                    detections = process_yolo_detections(detections_path)
                    self.detections_all = self.detections_all | detections
                
    def __len__(self):
        return len(self.rgb_frame_names)

    def __getitem__(self, idx):
        idx = int(idx)
        rgb_frame = cv2.imread(self.root_name + "/" + self.rgb_frame_names[idx])
        depth_frame = cv2.imread(self.root_name + "/" + self.depth_frame_names[idx], cv2.CV_16UC1)
        timestamp = self.timestamps[idx]

        if self.type == 'superpoint':
            points = self.points_all[self.rgb_frame_names[idx].split('/')[-1]]
            descriptor = self.descriptors_all[self.rgb_frame_names[idx].split('/')[-1]]
            sample = {'rgb': rgb_frame, 'depth': depth_frame,
                    'points': points, 'desc': descriptor,'timestamp':timestamp}
                    
        elif self.type == '3Dhist':
            key_names = self.rgb_frame_names[idx].split('/')[2].split('.')
            if (key_names[0] + '.' + key_names[1]) in self.detections_all.keys() and \
                len(self.detections_all[key_names[0] + '.' + key_names[1]]) > 8:
                detection = self.detections_all[key_names[0] + '.' + key_names[1]]
            else:
                detection = None
            sample = {'rgb': rgb_frame, 'depth': depth_frame,
                      'det': detection,'timestamp':timestamp}
        
        else:
            sample = {'rgb': rgb_frame, 'depth': depth_frame, 'timestamp':timestamp}

        return sample

def process_superpoint_feature_descriptors(superpoint_path: str) -> Tuple[dict, dict]:
    with open(superpoint_path + 'points.pickle', "rb") as f:
        points_all = pickle.load(f)
    with open(superpoint_path + 'descriptors.pickle', "rb") as f:
        descriptors_all = pickle.load(f)

    return points_all, descriptors_all

def process_yolo_detections(detections_path: str) -> dict:
    with open(detections_path + 'detection.pickle', "rb") as f:
        detections_all = pickle.load(f)
    return detections_all

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='''VDB fusion using known poses''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-v', '--visualize', default=True, type=bool, help='Visualize output')
    args = parser.parse_args()

    vdb_volume = VDBVolume(voxel_size=0.025, sdf_trunc=0.05, space_carving=False)

    dataset_name = 'apples_big_2021-10-14-all/'
    dataset = DatasetOdometryAll(args.data_root_path + dataset_name)

    frame_idx = np.loadtxt(f"../../eval_data/front/{dataset_name}superpoint/PnP/frames_min_dist_0.5_10deg.txt", np.int32)
    poses = np.loadtxt(f"../../eval_data/front/{dataset_name}superpoint/PnP/poses_min_dist_0.5_10deg.txt")

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    K = cam_intrinsics.intrinsic_matrix
    T = np.eye(4)

    T_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    T_rot_inv = np.linalg.inv(T_rot.T)

    for i, id in tqdm(enumerate(frame_idx[:-3])):
        rgbd = create_rgbd_from_color_and_depth(cv2.cvtColor(dataset[id]['rgb'], cv2.COLOR_BGR2RGB), dataset[id]['depth'])
        pcd = compute_pcl_from_rgbd(rgbd, cam_intrinsics, compute_normals = False)
        
        T[:3, :3] = o3d.geometry.PointCloud.get_rotation_matrix_from_quaternion(np.r_[poses[i, -1], poses[i, 4:-1]])
        T[:3, 3] = T_rot_inv @ poses[i, 1:4]

        pcd = pcd.transform(T)

        points = np.array(pcd.points)
        vdb_volume.integrate(points[:, :3], T[:3, 3])

    vertices, triangles = vdb_volume.extract_triangle_mesh()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"../../eval_data/front/{dataset_name}mesh.ply", mesh)
    if args.visualize:
        o3d.visualization.draw_geometries([mesh])