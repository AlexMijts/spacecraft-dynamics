import numpy as np
import sys
sys.path.append('../')

from attitude_coordinates.common.matrixOperations import compute_mat_products
from attitude_coordinates.euler_angles.eulerAngles import compute_euler321_to_dcm

def c2prv(c: np.array) -> tuple[float, np.ndarray]:
    '''
    Returns the principal rotation parameter set (rad)
    '''

    phi = np.arccos(0.5*(c[0,0]+ c[1,1]+c[2,2]-1))

    if (abs(phi) > np.pi):
        phi = phi - 2*np.pi

    e = (1/(2*np.sin(phi))) * np.array([c[1,2] - c[2,1], c[2,0] - c[0,2], c[0,1] - c[1,0]])

    return phi, e.T

if __name__ == "__main__":

    # Q2
    euler321 = np.deg2rad((20,-10,120))

    c = compute_euler321_to_dcm(euler321[0], euler321[1], euler321[2])

    rotation, prv = c2prv(c)
    print(rotation, prv)

    #Q3
    # dcm = np.array([[1, 0, 0],[0, 0, 1],[0, -1, 0]]) @ np.array([[1, 0, 0],[0, 0, 1],[0, -1, 0]])
    dcm = compute_mat_products(np.array([[1, 0, 0],[0, 0, 1],[0, -1, 0]]),
                               np.array([[1, 0, 0],[0, 0, 1],[0, -1, 0]]))

    print(dcm)
    rotation, prv = c2prv(dcm)
    print(rotation, prv)

    print(c2prv(dcm))


