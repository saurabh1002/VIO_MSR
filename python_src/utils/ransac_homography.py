#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.
    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
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
    

def compute_matches(D1: np.ndarray, D2: np.ndarray, h: int, w: int) -> (np.ndarray):
    """ Computes matches for two images using the descriptors, use the Lowe's criterea to determine the best match

    Arguments
    ---------
    - D1 : descriptors for image 1 corners
    - D2 : descriptors for image 2 corners

    Returns
    -------
    - M : [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints [num_matches x 2]
    """
    distances = cdist(D1, D2, 'euclidean')
    row_ind, col_ind = linear_sum_assignment(distances)

    M = np.array([row_ind, col_ind], np.uint8).T

    return M

def compute_homography_ransac(C1: np.ndarray, C2: np.ndarray, M: np.ndarray):
    """Implements a RANSAC scheme to estimate the homography and the set of inliers

    Arguments
    ---------
    C1 : numpy array [num_corners x 2]
         corner keypoints for image 1 
    C2 : numpy array [num_corners x 2]
         corner keypoints for image 2
    M  : numpy array [num_matches x 2]
        [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints 

    Returns
    -------
    H_final : numpy array [3 x 3]
            Homography matrix which maps in point image 1 to image 2 
    M_final : numpy array [num_inlier_matches x 2]
            [cornerIdx1, cornerIdx2] each row contains indices of inlier matches
    """
    # RANSAC parameters
    max_iter = 500
    min_inlier_ratio = 0.6
    inlier_thres = 2

    # num_matches = C1.shape[0]
    num_matches = M.shape[0]
    prev_best_num_inliers = 0

    for r in range(0, max_iter):
        # sample 4 matches and generate homograhic transformation matrix using these samples
        k = np.random.randint(0, num_matches, 4)
        p1 = C1[M[k, 0]]
        p2 = C2[M[k, 1]]
        H = calculate_homography_four_matches(p1, p2)
        residuals = compute_residual(C1[M[:, 0], :], C2[M[:, 1], :], H)
        # residuals = compute_residual(C1, C2, H)
        # Find number of inliers satisfying the inlier threshold
        num_of_inliers = np.sum(residuals < inlier_thres)
        # Indices of the inlier matches
        inliers_idx = np.where(residuals < inlier_thres)
        # Save the inliers if they are better than the previous iteration
        if num_of_inliers > prev_best_num_inliers:
            prev_best_num_inliers = num_of_inliers
            H_final = H
            M_final = M[inliers_idx[0], :]
        # If inlier ratio criteria is met, then exit the loop and return the latest inliers
        if num_of_inliers > (min_inlier_ratio * num_matches):
            H_final = H
            M_final = M[inliers_idx[0], :]
            break

    return H_final, M_final#prev_best_num_inliers 

# Calculate the geometric distance between estimated points and original points, namely residuals.
def compute_residual(P1: np.ndarray, P2: np.ndarray, H: np.ndarray) -> (np.ndarray):
    """
    Compute the residual given the Homography H

    Arguments
    ---------
    - P1: Points (x,y) from Image 1 [num_points x 2]
    - P2: Points (x,y) from Image 2 [num_points x 2]  
    - H: Homography which maps P1 to P2 [3 x 3]

    Returns
    -------
    - residuals : residual computed for the corresponding points P1 and P2 under the transformation given by H          
    """
    # Compute the predicted coordinates of keypoints in first image warped by the homography
    # print(H)
    # print(P1_homo)
    P1_homo = np.hstack((P1, np.ones((P1.shape[0], 1)))).T
    P2_pred_homo = H @ P1_homo
    P2_pred = P2_pred_homo[:2, :] / P2_pred_homo[2, :].reshape(1, -1)
    P2_pred = P2_pred.T

    # Find the error between the predicted and actual keypoint coordinates in image 2
    diff = P2_pred - P2
    residuals = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

    return residuals

def calculate_homography_four_matches(P1: np.ndarray, P2: np.ndarray) -> (np.ndarray):
    """ Estimate the homography given four correspondening keypoints in the two images

    Arguments
    ---------
    - P1: Points (x, y) from Image 1 [num_points x 2]

    - P2: Points (x, y) from Image 2 [num_points x 2]
    
    Returns
    -------
    - H: Homography which maps P1 to P2 based on the four corresponding points [3 x 3]
    """
    if P1.shape[0] != 4 or P2.shape[0] != 4:
        print('Four corresponding points needed to compute Homography')
        return None

    # loop through correspondences and create assemble matrix
    # A * h = 0, where A(2n,9), h(9,1)
    A = []
    for i in range(P1.shape[0]):
        p1 = np.array([P1[i, 0], P1[i, 1], 1])
        p2 = np.array([P2[i, 0], P2[i, 1], 1])

        a2 = [
            0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
            p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]
        ]
        a1 = [
            -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
            p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]
        ]
        A.append(a1)
        A.append(a2)

    A = np.array(A)

    # svd composition
    u, s, v = np.linalg.svd(A)
    H = np.reshape(v[8], (3, 3))

    # normalize and now we have H
    H = (1 / H[2, 2]) * H

    return H
