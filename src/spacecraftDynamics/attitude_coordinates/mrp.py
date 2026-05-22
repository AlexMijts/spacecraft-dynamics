import numpy as np
import sys
sys.path.append('../')

from attitude_coordinates.common.matrixOperations import compute_mat_products
from Basilisk.utilities import RigidBodyKinematics as rbk

def compute_shadow_set(s: np.array) -> np.array:

    s_shadow = np.array([0.0, 0.0, 0.0])
    div = 1/(np.linalg.norm(s)**2)

    for i in range(len(s)):
        s_shadow[i] = -div * s[i]

    return s_shadow

def compute_dcm_to_mrp(c : np.array) -> np.array:
    csi_fact = 1/( np.sqrt(np.trace(c)+1) * (np.sqrt(np.trace(c)+1)+2) )
    s = np.array([c[1,2]-c[2,1], c[2,0]-c[0,2], c[0,1]-c[1,0]])

    return (csi_fact*s)

def mrp_to_dcm(sigma) -> np.ndarray:
    # Ensure it's a numpy array for vector operations
    sigma = np.asarray(sigma, dtype=float)

    # Calculate the squared magnitude of the MRP vector
    norm_2 = np.dot(sigma, sigma)

    # Create the skew-symmetric matrix [sigma_tilde]
    s_tilde = np.array([
        [0,        -sigma[2],  sigma[1]],
        [sigma[2],  0,        -sigma[0]],
        [-sigma[1], sigma[0],  0]
    ])

    # Common denominator term to simplify the math
    denominator = (1 + norm_2) ** 2

    # Construct the DCM
    I = np.eye(3)
    term1 = (4 * (1 - norm_2) / denominator) * s_tilde
    term2 = (8 / denominator) * (s_tilde @ s_tilde)

    dcm = I + term1 + term2

    return dcm

if __name__ == "__main__":

    print("# ------- CONCEPT CHECK 17 -------#")
    print("------Q1------")
    sigma = np.array([0.1,0.2,0.3])
    sigma_s = compute_shadow_set(sigma)
    print(sigma_s)

    print("# ------- CONCEPT CHECK 18 -------#")
    print("------Q1------")
    print(rbk.MRP2C(sigma))

    print("------Q2------")
    dcm = np.array([
        [0.763314, 0.0946746, -0.639053],
        [-0.568047,-0.372781, -0.733728],
        [-0.307692,  0.923077, -0.230769]
    ])
    s = rbk.C2MRP(dcm)

    print(s)
    print(compute_dcm_to_mrp(dcm))

    print("# ------- CONCEPT CHECK 19 -------#")
    print("------Q2------")
    sigma1 = np.array([0.1,0.2,0.3])
    sigma2 = np.array([-0.1,0.3,0.1])
    print(rbk.addMRP(sigma1, sigma2))

    print("------Q3------")
    sigma1 = np.array([0.1,0.2,0.3])
    sigma2 = np.array([0.5,0.3,0.1])
    print(rbk.addMRP(-sigma2, sigma1))


