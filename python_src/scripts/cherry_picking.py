import os

import sys; sys.path.append(os.pardir)
import argparse
from tqdm import tqdm

from utils.icp import *
from utils.utils import *

from descriptor_3d_hist import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.dataloader import DatasetOdometry, DatasetOdometryAll
from vo_odom_pnp import getMatches
from utils.ransac_homography import compute_homography_ransac
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''VO using PnP algorithm with Superpoint Features''')

    # Dataset paths
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/right/',
                        type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")


    args = parser.parse_args()
    skip = args.skip_frames
    method = ['superpoint','orb','3Dhist']

    dataset_name = 'apples_big_2021-10-14-14-53-32_4/'
#    apples_big_2021-10-14-14-51-44_1
#    apples_big_2021-10-14-14-52-20_2
#    apples_big_2021-10-14-14-52-56_3
#    apples_big_2021-10-14-14-53-32_4
    
    dataset_orb =DatasetOdometry(args.data_root_path + dataset_name, method[1])
    dataset_3d = DatasetOdometry(args.data_root_path + dataset_name, method[2])
    dataset_sp = DatasetOdometry(args.data_root_path + dataset_name, method[0])
    # idx = np.loadtxt('wrong_detection.txt').astype(np.int64)
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    
    if "right" in args.data_root_path:
        max_depth = 5
        cam_intrinsics.set_intrinsics(640, 480, 606.6, 605.4, 323.2, 247.4)
    elif "front" in args.data_root_path:
        max_depth = 10
        cam_intrinsics.set_intrinsics(640, 480, 381.5, 381.2, 315.5, 237.8)


    K = cam_intrinsics.intrinsic_matrix.astype(np.double)
    
    num_matches_sp  = np.zeros((len(dataset_3d)-1))
    num_matches_3d  = np.zeros((len(dataset_3d)-1))
    num_matches_orb = np.zeros((len(dataset_3d)-1))

    num_inlier_sp = np.zeros((len(dataset_3d)-1))
    num_inlier_3d = np.zeros((len(dataset_3d)-1))
    num_inlier_orb = np.zeros((len(dataset_3d)-1))
    # wront_detection = []
    # for i in idx:

    for i in tqdm(range(len(dataset_3d)-skip)):
   
    
        pts_3d1, pts_3d2,_,_,det = getMatches(dataset_3d[i],dataset_3d[i+skip],type='3Dhist',K=K)
        pts_orb1, pts_orb2,_,_,_ = getMatches(dataset_orb[i],dataset_orb[i+skip],type='ORB',K=K)
        pts_sp1,pts_sp2,_,_,_ =    getMatches(dataset_sp[i],dataset_sp[i+skip],type='superpoint',K=K)
        # pts_sp1 = pts_sp1[:200]
        # pts_sp2 = pts_sp2[:200]
        if not det:
            continue
        
    #     if not det or pts_3d1.shape[0] < 4 or pts_3d2.shape[0] < 4:
    #         if not det:
    #             num_matches_3d[i] = 1
    #         else:
    #             num_matches_3d[i] = pts_3d1.shape[0]
    #         num_inlier_3d[i] = 0
    #     else:
    #         num_matches_3d[i] = pts_3d1.shape[0]
    #         _, inlier_3d = compute_homography_ransac(pts_3d1,pts_3d2, None)
    #         num_inlier_3d[i] = inlier_3d
            
    #     _, inlier_sp = compute_homography_ransac(pts_sp1,pts_sp2, None)
    #     _, inlier_orb = compute_homography_ransac(pts_orb1,pts_orb2, None)
        
    #     num_matches_sp[i] = pts_sp1.shape[0]
    #     num_matches_orb[i] = pts_orb1.shape[0]

    #     num_inlier_sp[i] = inlier_sp
    #     num_inlier_orb[i] =inlier_orb
    
    # avg_inlier_ratio_3d = np.average(np.divide(num_inlier_3d,num_matches_3d))
    # avg_inlier_ratio_orb = np.average(np.divide(num_inlier_orb,num_matches_orb))
    # avg_inlier_ratio_sp = np.average(np.divide(num_inlier_sp,num_matches_sp))
    
    # avg_matches_3d = np.average(num_matches_3d)
    # avg_matches_orb = np.average(num_matches_orb)
    # avg_matches_sp = np.average(num_matches_sp)
    
   
    # print(f"Inlier Ratio : SP : {avg_inlier_ratio_sp}, 3d : {avg_inlier_ratio_3d}, ORB : {avg_inlier_ratio_orb}")
    # print(f"Matches : SP : {avg_matches_sp}, 3d : {avg_matches_3d}, ORB : {avg_matches_orb}")
        
        # Images

        img1 = cv2.cvtColor(dataset_orb[i]['rgb'].astype('float32') / 255.0,cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(dataset_orb[i+skip]['rgb'].astype('float32') / 255.0,cv2.COLOR_RGB2GRAY)

        img1_rgb = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        img2_rgb = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)

        # img_sp = dataset_sp[i]['rgb']
        # img_orb = dataset_sp[i]['rgb']
        w = 640
        h = 480

        rgb_match_frame_orb  = np.concatenate((img1_rgb, img2_rgb), 1)
        # rgb_match_frame_3dhist  = np.concatenate((img1_rgb, img2_rgb), 1)
        # rgb_match_frame_sp  = np.concatenate((img1_rgb, img2_rgb), 1)


        color_line = np.array([3,252,136]) / 255.0
        # color_point = np.array([3,252,136]) / 255.0
        color_point = np.array([0,0,255]) / 255.0

        for kp in pts_orb1:
            cv2.circle(rgb_match_frame_orb, (int(kp[0]), int(kp[1])), 3, (color_point[0],color_point[1],color_point[2]), -1)

        for kp in pts_orb2:
            cv2.circle(rgb_match_frame_orb, (int(kp[0]) + w,int(kp[1])), 3, (color_point[0],color_point[1],color_point[2]), -1)

        for kp_l, kp_r in zip(pts_orb1.astype(np.int64), pts_orb2.astype(np.int64)):
            cv2.line(rgb_match_frame_orb, (kp_l[0], kp_l[1]), (kp_r[0] + w , kp_r[1]), (color_line[0],color_line[1],color_line[2]), 2)

        # for kp in pts_3d1:
        #     cv2.circle(rgb_match_frame_3dhist, (int(kp[0]), int(kp[1])), 3, (color_point[0],color_point[1],color_point[2]), -1)

        # for kp in pts_3d2:
        #     cv2.circle(rgb_match_frame_3dhist, (int(kp[0]) + w,int(kp[1])), 3, (color_point[0],color_point[1],color_point[2]), -1)

        # for kp_l, kp_r in zip(pts_3d1.astype(np.int64), pts_3d2.astype(np.int64)):
        #     cv2.line(rgb_match_frame_3dhist, (kp_l[0], kp_l[1]), (kp_r[0] + w , kp_r[1]), (color_line[0],color_line[1],color_line[2]), 2)

        # for kp in pts_sp1:
        #     cv2.circle(img_sp, (int(kp[0]), int(kp[1])), 3, (color_point[0],color_point[1],color_point[2]), -1)

        # for kp in pts_sp1:
        #     cv2.circle(rgb_match_frame_sp, (int(kp[0]) + w,int(kp[1])), 3, (color_point[0],color_point[1],color_point[2]), -1)

        # for kp_l, kp_r in zip(pts_sp1.astype(np.int64), pts_sp2.astype(np.int64)):
        #     cv2.line(rgb_match_frame_sp, (kp_l[0], kp_l[1]), (kp_r[0] + w , kp_r[1]), (color_line[0],color_line[1],color_line[2]), 2)

    
        # frame_all = np.vstack((rgb_match_frame_orb,rgb_match_frame_3dhist))

        # frame_all = np.vstack((rg))

        cv2.imshow("output", rgb_match_frame_orb)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # close on ESC key
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):

            # cv2.imwrite(f'../../eval_data/wrong_detection/keypoints_sp_{i}.png',rgb_match_frame_sp*255.0)
            
            # cv2.imwrite(f'../../eval_data/wrong_detection/detection_10_sp_{i}.png',rgb_match_frame_sp*255.0)
            cv2.imwrite(f'../../eval_data/wrong_detection/motivation_orb_{i}.png',rgb_match_frame_orb*255)
            # cv2.imwrite(f'../../eval_data/wrong_detection/detection_10_3dhist_{i}.png',rgb_match_frame_3dhist*255)

    # cv2.destroyAllWindows()
    # np.savetxt('wrong_detection.txt',np.asarray(wront_detection))