#!/usr/bin/env python

# ==============================================================================
# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de
# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
from numpy import linalg as la

def compute_W(Q: np.ndarray, P: np.ndarray, MuQ: np.ndarray, MuP: np.ndarray) -> (np.ndarray):    
    """ Compute cross covariance matrix W to use in SVD
    
    Arguments
    ---------
    - Q: Reference Pointcloud [M x N] matrix of N points in M dimensions
    - P: Pointcloud to be transformed [M x N] matrix of N points in M dimensions
    - MuQ: [M x 1] Mean of points in Pointcloud Q
    - MuP: [M x 1] Mean of points in Pointcloud P
    
    Returns
    -------
    - W: [M x M] Cross covariance matrix
    """
    
    # Mean reduced Pointclouds
    Q_0_mean = Q - MuQ
    P_0_mean = P - MuP

    # Calculate cross covariance matrix W
    W = np.zeros((np.shape(P)[0], np.shape(P)[0]))

    for i in range(len(Q[0])):
        an = P_0_mean[:, i].reshape((np.shape(P)[0], 1))
        bn = Q_0_mean[:, i].reshape((np.shape(Q)[0], 1))
        W = W + (an @ bn.T)

    return W

def compute_R_t(W: np.ndarray, MuQ: np.ndarray, MuP: np.ndarray) -> tuple[np.ndarray, np.ndarray]:    
    """ Compute rotation matrix and translation vector based on the SVD

    Arguments
    ---------
    - W: [M x M] Cross covariance of mean reduced pointclouds
    - MuQ: [M x 1] Mean of points in Pointcloud Q
    - MuP: [M x 1] Mean of points in Pointcloud P
    
    Returns
    -------
    - R: [M x M] Rotation matrix in M dimensions
    - t: [M x 1] Translation vector in M dimensions
    """
    U, D, Vt = la.svd(W)
    R = Vt.T @ U.T
    t = MuQ - R @ MuP

    return R, t

def icp_known_corresp(P1: np.ndarray, P2: np.ndarray, QInd: np.ndarray, PInd: np.ndarray) -> (np.ndarray):
    """ Performs ICP alignment between two Pointclouds with known correspondances

    Arguments
    ---------
    - P1: Reference Pointcloud [M x N] matrix of N points in M dimensions
    - P2: Pointcloud to be transformed [M x N] matrix of N points in M dimensions
    - QInd: [N x 1] Correspondance Indices list for Line1
    - PInd: [N x 1] Correspondance Indices list for Line2
    
    Returns
    -------
    - T: [M+1 x M+1] homogenous transformation matrix
    """
    # Arrange Pointclouds in order of correspondances as given by QInd and PInd
    Q = P1[:, QInd]
    P = P2[:, PInd]
    # Compute mean of the two Pointclouds
    MuQ = np.mean(Q, 1).reshape(np.shape(Q)[0], 1)
    MuP = np.mean(P, 1).reshape(np.shape(P)[0], 1)
    
    # Compute cross covariance matrix
    W = compute_W(Q, P, MuQ, MuP)

    # Compute Rotation and translation
    [R, t] = compute_R_t(W, MuQ, MuP)
   
    # Compute the new positions of the points after
    # applying found rotation and translation to them
    T = np.eye(P1.shape[0] + 1)
    T[:-1, :-1] = R
    T[:-1, -1] = t.reshape(-1,)

    return la.inv(T)