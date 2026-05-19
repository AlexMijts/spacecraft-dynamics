import numpy as np
import sys
sys.path.append('../')

from Basilisk.utilities import RigidBodyKinematics as rbk

def crp_to_dcm(q: np.array)-> np.ndarray:

    q1= q[0]
    q2= q[1]
    q3= q[2]
    fact = 1/(1 + np.linalg.norm(q)**2)
    c = np.array([[ 1 + q1**2 - q2**2 - q3**2, 2*(q1*q2+q3), 2*(q1*q3-q2)  ],
                    [  2*(q1*q2-q3), 1 - q1**2 + q2**2 - q3**2, 2*(q2*q3+q1) ],
                    [  2*(q1*q3+q2),  2*(q2*q3-q1), 1 - q1**2 - q2**2 + q3**2] ])

    return fact*c

def dcm_to_crp(c : np.ndarray) -> np.ndarray:
    c23 = c[1][2]
    c32 = c[2][1]
    c31 = c[2][0]
    c13 = c[0][2]
    c12 = c[0][1]
    c21 = c[1][0]
    csi_2_inv = 1/(np.sqrt(np.trace(c)+1)**2)
    q = np.array([c23-c32, c31-c13, c12-c21])

    return csi_2_inv*q

def add_crp(q1 : np.array, q2 : np.array) -> np.array:
    return (    1/(1 - np.dot(q2,q1)) * ((q2 + q1) - np.cross(q2,q1))    )

def subtract_crp(q1 : np.array, q2 : np.array) -> np.array:
    return (    1/(1 + np.dot(q1,q2)) * ((q1 - q2) + np.cross(q1,q2))    )

if __name__ == '__main__':

    #********************** Concept Check 11,12 **********************#
    print('Q1')
    q = np.array([0.1, 0.2, 0.3])
    print(crp_to_dcm(q))

    print('Q2')
    dcm = np.array([[0.333333, -0.666667, 0.666667],
                   [0.871795, 0.487179, 0.0512821],
                   [-0.358974, 0.564103, 0.74359]])
    print(dcm_to_crp(dcm))

    print('Q4')
    print(subtract_crp(np.array([-0.3, 0.3, 0.1]), q))



