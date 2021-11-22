#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta ; Dhagash Desai
# @email: s7sagupt@uni-bonn.de ; s7dhdesa@uni-bonn.de

# MSR Project Sem 3: Visual Inertial Odometry in Orchard Environments

# This script defines some utility functions that are used often in other scripts

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
from typing import NoReturn


def read_file(path: str, delimiter: str = ' ') -> (np.ndarray):
    """ Read data from a file

    Arguments
    ---------
    - path: Path of ASCII file to read
    - delimiter: Delimiter character for the file to be read

    Returns
    -------
    - data: Data from file as a numpy array
    """
    data = np.loadtxt(path, delimiter=delimiter)

    return data


def wrapTo2Pi(theta: float) -> (float):
    """
    Wrap around angles to the range [0, 2pi]
    
    Arguments
    ---------
    - theta: angle
    Returns
    -------
    - theta: angle within range [0, 2pi]
    """
    while theta < 0:
        theta += 2 * np.pi
    while theta > 2 * np.pi:
        theta -= 2 * np.pi
    return theta