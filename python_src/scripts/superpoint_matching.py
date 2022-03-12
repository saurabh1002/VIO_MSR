import os
import sys
sys.path.append(os.pardir)
import argparse
from tqdm import tqdm
from typing import Tuple

from utils.dataloader import DatasetOdometry, DatasetOdometryAll

import cv2
import numpy as np

def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    
    Arguments
    ---------
    - desc1: NxM numpy matrix of N corresponding M-dimensional descriptors.
    - desc2: NxM numpy matrix of N corresponding M-dimensional descriptors.
    - nn_thresh: Optional descriptor distance below which is a good match.
    
    Returns
    -------
    - matches: 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches[:2].T

def getMatches(keypts_1: dict, keypts_2: dict, threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Get Matching SuperPoint Features
    
    Arguments
    ---------
    - keypts_1: Keypoints in image frame 1
    - keypts_2: Keypoints in image frame 2
    - threshold: Two way matching threshold

    Returns
    -------
    - points2d_1: Matched 2D keypoints in frame 1
    - points2d_2: Matched 2D keypoints in frame 2
    """
    points2d_1 = keypts_1['points'][:,:2]
    points2d_2 = keypts_2['points'][:,:2]

    descriptor_1 = keypts_1['desc']
    descriptor_2 = keypts_2['desc']

    M = nn_match_two_way(descriptor_1.T, descriptor_2.T, threshold).astype(np.int64)

    return points2d_1[M[:, 0]], points2d_2[M[:, 1]]

def argparser():
    """Argument Parser
    """
    parser = argparse.ArgumentParser(description='''Matches Descriptors from the SuperPoint Algorithm''')
    parser.add_argument('-i', '--data_root_path', default='../../datasets/phenorob/front/', type=str,
        help='Path to the root directory of the dataset')
    parser.add_argument('-n', '--skip_frames', default=1, type=int, help="Number of frames to skip")
    parser.add_argument('-v', '--visualize', default=False, type=bool, help='Visualize results')
    parser.add_argument('-s', '--save', default=False, type=bool, help='Save flag')
    
    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = argparser()
    dataset_name = 'apples_big_2021-10-14-all/'

    if not "all" in dataset_name:
        dataset = DatasetOdometry(args.data_root_path + dataset_name)
    else:
        dataset = DatasetOdometryAll(args.data_root_path + dataset_name)

    cam_dir = 'front' if 'front' in args.data_root_path else 'right'
    skip = args.skip_frames
    progress_bar = tqdm(range(0, len(dataset) - skip, skip))
    for i in progress_bar:
        keypts_2d_1, keypts_2d_2 = getMatches(dataset[i], dataset[i + skip])

        if args.visualize or args.save:            
            w = 640
            rgb_match_frame = np.concatenate((dataset[i]['rgb'], dataset[i + skip]['rgb']), 1)
            for kp in keypts_2d_1:
                cv2.circle(rgb_match_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

            for kp in keypts_2d_2:
                cv2.circle(rgb_match_frame, (int(kp[0]) + w,
                        int(kp[1])), 3, (0, 0, 255), -1)
            output_frame = np.copy(rgb_match_frame)
        
            for kp_l, kp_r in zip(keypts_2d_1[:50].astype(int), keypts_2d_2[:50].astype(int)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            for kp_l, kp_r in zip(keypts_2d_1.astype(int), keypts_2d_2.astype(int)):
                cv2.line(rgb_match_frame, (kp_l[0], kp_l[1]), (kp_r[0] + w, kp_r[1]), (0, 255, 255), 2)
            output_frame = np.concatenate((output_frame, rgb_match_frame), 0)
            if args.visualize:
                cv2.imshow("output", output_frame)
                cv2.waitKey(0)
            if args.save:
                cv2.imwrite(
                    f'../../eval_data/{cam_dir}/{dataset_name}superpoint/{int(i / skip)}.png', output_frame)
    
    cv2.destroyAllWindows()