import numpy as np
import sys
sys.path.append('../')

from .common.matrixOperations import compute_mat_products

def compute_euler321_to_dcm(theta1: float, theta2: float, theta3: float):
    """
    Computes the direction cosine matrix (DCM) for a sequence of three rotations
    using the ZYX (yaw-pitch-roll) Tait-Bryan angle convention.

    Args:
        theta1 (float): Rotation angle about the x-axis (roll) in radians.
        theta2 (float): Rotation angle about the y-axis (pitch) in radians.
        theta3 (float): Rotation angle about the z-axis (yaw) in radians.

    Returns:
        np.ndarray: The resulting 3x3 DCM as a NumPy array.
    """
    angle = theta1
    c = np.cos(angle)
    s = np.sin(angle)
    m1 = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])

    angle = theta2
    c = np.cos(angle)
    s = np.sin(angle)
    m2 = np.array([[c, 0, -s],
                   [0, 1, 0],
                   [s, 0, c]])

    angle = theta3
    c = np.cos(angle)
    s = np.sin(angle)
    m3 = np.array([[1, 0, 0],
                   [0, c, s],
                   [0, -s, c]])

    output = compute_mat_products(m3, m2, m1)

    return output


def compute_dcm_to_euler313(mat: np.ndarray) -> tuple:
    """
    Extracts the set of three proper Euler angles (in radians) from a 3x3 rotation matrix
    using the ZXZ 'proper euler angle' convention.

    Args:
        mat (np.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: (alpha, beta, gamma) angles in radians.
    """
    if mat.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")

    # ZXZ Euler angles extraction:
    # beta = arccos(r33)
    beta = np.arccos(mat[2, 2])

    # Check for gimbal lock
    if np.isclose(beta, 0):
        # beta == 0, alpha + gamma = atan2(r12, r11)
        alpha = 0
        gamma = np.arctan2(mat[0, 1], mat[0, 0])
    elif np.isclose(beta, np.pi):
        # beta == pi, alpha - gamma = atan2(-r12, -r11)
        alpha = 0
        gamma = np.arctan2(-mat[0, 1], -mat[0, 0])
    else:
        alpha = np.arctan2(mat[2, 0], -mat[2, 1])
        gamma = np.arctan2(mat[0, 2], mat[1, 2])

    return (alpha, beta, gamma)


def compute_dcm_to_euler321(mat: np.ndarray) -> tuple:
    """
    Extracts the set of three Euler angles (in radians) from a 3x3 rotation matrix
    using the ZYX (yaw-pitch-roll) convention.

    Args:
        mat (np.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: (psi, theta, phi) yaw-pitch-roll angles in radians.
    """
    if mat.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3.")

    # ZYX Euler angles extraction with updated r values:
    # r11 = mat[0, 0], r12 = mat[0, 1], r13 = mat[0, 2]
    # r23 = mat[1, 2], r33 = mat[2, 2]
    theta = np.arcsin(-mat[0, 2])  # theta = arcsin(-r13)
    if np.abs(mat[0, 2]) != 1:
        psi = np.arctan2(mat[0, 1], mat[0, 0])   # psi = atan2(r12, r11)
        phi = np.arctan2(mat[1, 2], mat[2, 2])   # phi = atan2(r23, r33)
    else:
        # Gimbal lock
        phi = 0
        if mat[0, 2] == -1:
            theta = np.pi / 2
            psi = np.arctan2(mat[1, 0], mat[1, 1])
        else:
            theta = -np.pi / 2
            psi = np.arctan2(-mat[1, 0], -mat[1, 1])
    return (psi, theta, phi)

if __name__ == "__main__":

    #********************** Concept Check 7 **********************#

    # Angle set
    theta7_deg = (20, 10, -10)
    theta7_rad = np.deg2rad(theta7_deg)

    # Compute DCM based on ZYX Tait-Bryan Euler set
    DCM_7 = compute_euler321_to_dcm(theta7_rad[0], theta7_rad[1], theta7_rad[2])

    # Extract corresponding ZXZ proper euler angle set based on DCM values
    angles = compute_dcm_to_euler313(DCM_7)

    #********************** Concept Check 8 **********************#

    theta8_deg = np.array([10, 20, 30]) #yaw-Z, pitch-Y, roll-X
    theta8_rad = np.deg2rad(theta8_deg)

    # Compute DCM based on ZYX Tait-Bryan Euler set
    DCM_8 = compute_euler321_to_dcm(theta8_rad[0], theta8_rad[1], theta8_rad[2])

    # Extract corresponding ZXZ proper euler angle set based on DCM values
    angles = compute_dcm_to_euler313(DCM_8)
    print(np.rad2deg(angles))

    BN = compute_euler321_to_dcm(np.deg2rad(10), np.deg2rad(20), np.deg2rad(30))
    RN = compute_euler321_to_dcm(np.deg2rad(-5), np.deg2rad(5), np.deg2rad(5))
    BR = BN @ RN.T
    bn_ea = compute_dcm_to_euler321(BR)

    print(np.rad2deg(bn_ea))
