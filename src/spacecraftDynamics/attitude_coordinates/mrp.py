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


