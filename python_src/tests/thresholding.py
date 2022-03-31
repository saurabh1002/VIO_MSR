import os
import pickle
import argparse
from tqdm import tqdm
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


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


parser = argparse.ArgumentParser(description='''Colorspace Thresholding for apple detection''')
parser.add_argument('-b', '--bboxes_path', default='../../datasets/phenorob/images_apples_right/detection.pickle', type=str,
    help='Path to the centernet object detection bounding box coordinates')
parser.add_argument('-a', '--associations_path', default='../../datasets/phenorob/images_apples_right/associations_rgbd.txt', type=str,
    help='Path to the associations file for RGB and Depth frames')
parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/images_apples_right/', type=str,
    help='Path to the root directory of the dataset')

parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
args = parser.parse_args()

bboxes_d, rgb_names, depth_names = process_input_data(args.bboxes_path, args.associations_path)
num_of_frames = len(rgb_names)

rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
if args.data_root_path == "../../datasets/phenorob/images_apples_right/":
    rgb_camera_intrinsic.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
elif args.data_root_path == '../../datasets/phenorob/images_apples/':
    rgb_camera_intrinsic.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)
    

skip_frames = 10
for n in tqdm(range(0, num_of_frames - skip_frames, skip_frames)):

    img_rgb = cv2.imread(args.data_root_path + rgb_names[n])
    img_depth = cv2.imread(args.data_root_path + depth_names[n], cv2.CV_16UC1)

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Chromatic abberence and chromatic abberence ratio based thresholding
    rgb_mask = (img_rgb[:, :, 0] > img_rgb[:, :, 1]) & (img_rgb[:, :, 2] > img_rgb[:, :, 1])

    # HSV image, saturation value Otsu thresholding
    _, hsv_mask = cv2.threshold(img_hsv[:, :, 1], 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    hsv_masked = np.multiply(img_hsv, np.dstack((hsv_mask, hsv_mask, hsv_mask)))
    rgb_masked = np.multiply(img_rgb, np.dstack((rgb_mask, rgb_mask, rgb_mask)))

    kernel = np.ones((5,5),np.uint8)
    mask = rgb_mask & hsv_mask
    
    rgb_masked = np.multiply(img_rgb, np.dstack((mask, mask, mask)))
    depth_masked = np.multiply(img_depth, mask)

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax1.imshow(img_rgb)
    # ax2.imshow(hsv_masked, cmap='hsv')
    # ax3.imshow(final_mask)
    # ax4.imshow(final_masked)
    # plt.pause(-1)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_masked), 
            o3d.geometry.Image(depth_masked), 
            depth_scale=1000, depth_trunc=3, 
            convert_rgb_to_intensity=False)

    target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, rgb_camera_intrinsic)

    o3d.visualization.draw_geometries([target])

    show_contours = False
    if (show_contours):
        contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        cont_area_threshold = 100
        thresholded_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > cont_area_threshold:
                thresholded_contours.append(contour)
        image_cont = img_rgb.copy()
        cv2.drawContours(image=image_cont, 
                         contours=thresholded_contours, 
                         contourIdx=-1, 
                         color=(0, 255, 0), 
                         thickness=2, 
                         lineType=cv2.LINE_AA)
        # see the results
        plt.imshow(image_cont)
        plt.pause(-1)
        plt.close()

    show_hist = False
    if show_hist:
        hist_lab_l, _ = np.histogram(img_lab[:, :, 0], 256)
        hist_lab_a, _ = np.histogram(img_lab[:, :, 1], 256)
        hist_lab_b, _ = np.histogram(img_lab[:, :, 2], 256)

        hist_hsv_h, _ = np.histogram(img_hsv[:, :, 0], 256)
        hist_hsv_s, _ = np.histogram(img_hsv[:, :, 1], 256)
        hist_hsv_v, _ = np.histogram(img_hsv[:, :, 2], 256)

        hist_bgr_r, _ = np.histogram(img_rgb[:, :, 0], 256)
        hist_bgr_g, _ = np.histogram(img_rgb[:, :, 1], 256)
        hist_bgr_b, _ = np.histogram(img_rgb[:, :, 2], 256)

        plt.plot(hist_bgr_b)
        plt.plot(hist_bgr_g)
        plt.plot(hist_bgr_r)
        plt.plot(hist_lab_l)
        plt.plot(hist_lab_a)
        plt.plot(hist_lab_b)
        plt.plot(hist_hsv_h)
        plt.plot(hist_hsv_s)
        plt.plot(hist_hsv_v)
        plt.legend(['b', 'g', 'r', 'l', 'a', 'b', 'h', 's', 'v'])
        plt.ion()
        plt.pause(-1)