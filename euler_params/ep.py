import numpy as np
import sys
sys.path.append('../')

from euler_angles.eulerAngles import compute_euler321_to_dcm
from Basilisk.utilities import RigidBodyKinematics as rbk

def ep2c(beta: np.array) -> np.ndarray:
    '''
    return the DCM corresponding to the input quaternion
    '''

    beta0 = beta[0]
    beta1 = beta[1]
    beta2 = beta[2]
    beta3 = beta[3]

    c = np.array([
        [beta0**2+beta1**2-beta2**2-beta3**2, 2 * (beta1*beta2 - beta0*beta3), 2 * (beta1*beta3 + beta0*beta2)],
        [2 * (beta1*beta2 + beta0*beta3), beta0**2-beta1**2+beta2**2+beta3**2, 2 * (beta2*beta3 - beta0*beta1)],
        [2 * (beta1*beta3 - beta0*beta2), 2 * (beta2*beta3 + beta0*beta1), beta0**2-beta1**2-beta2**2+beta3**2]
    ])

    return c.T

def c2ep(c: np.ndarray) -> np.ndarray:
    '''
    return the quaternion corresponding to the input DCM
    '''
    beta0 = 0.5 * np.sqrt(1 + c[0,0] + c[1,1] + c[2,2])
    beta1 = (c[1,2] - c[2,1]) / (4 * beta0)
    beta2 = (c[2,1] - c[1,2]) / (4 * beta0)
    beta3 = (c[0,1] - c[1,0]) / (4 * beta0)

    beta = np.array([beta0, beta1, beta2, beta3])

    return beta

def addEp(q1: np.array, q2: np.array) -> np.ndarray:
    '''
    Computes the rotation corresponding to the successive rotation of q2 and q1
    '''
    a = np.array([[q1[0], -q1[1], -q1[2], -q1[3]],
                  [q1[1],  q1[0],  q1[3], -q1[2]],
                  [q1[2], -q1[3],  q1[0],  q1[1]],
                  [q1[3],  q1[2], -q1[1],  q1[0]]])
    res = a @ np.array([[q2[0]], [q2[1]], [q2[2]], [q2[3]]])

    return res.T

if __name__ == "__main__":

    print("# ------- CONCEPT CHECK 6 -------#")
    # Q1
    print("------Q1------")
    beta = np.array([0.235702,0.471405,-0.471405,0.707107])

    print(f"My ans : {ep2c(beta)}")
    print(f"RBK ans : {rbk.EP2C(beta)}")

    # Q3
    print("------Q3------")
    dcm_manual = np.array([
        [-0.529403,-0.467056,0.708231],
        [-0.474115,-0.529403,-0.703525],
        [0.703525,-0.708231,0.0588291]
    ])

    ep = c2ep(dcm_manual)
    print(f"My ans : {ep}" )
    print(f"RBK ans : {rbk.C2EP(dcm_manual)}")

    # Q4
    print("------Q4------")
    dcm_3 = compute_euler321_to_dcm(np.deg2rad(20),np.deg2rad(10),np.deg2rad(-10))
    dcm3rbk = rbk.euler3212C(np.deg2rad((20,10,-10)))

    print(f"My ans : {c2ep(dcm_3)}")
    print(f"RBK ans : {rbk.C2EP(dcm3rbk)}")

    print("# ------- CONCEPT CHECK 7 -------#")
    print("------Q1------")
    beta_fb = (0.359211, 0.898027, 0.179605, 0.179605)
    beta_bn = (0.774597, 0.258199, 0.516398, 0.258199)

    print(f"My ans : {addEp(beta_fb, beta_bn)}")
    print(f"rbk ans : {rbk.addEP(beta_bn, beta_fb)}")
    dcm = rbk.EP2C(beta_fb) @ rbk.EP2C(beta_bn)
    print(rbk.C2EP(dcm))

    print("------Q2------")
    beta_fn = (0.359211, 0.898027, 0.179605, 0.17605)
    beta_bn = (-0.377964, 0.755929, 0.377964, 0.377964)

    beta_nb = (beta_bn[0], -beta_bn[1], -beta_bn[2], -beta_bn[3])
    print(f"rbk ans : {rbk.addEP(beta_nb, beta_fn)}")

    dcm = rbk.EP2C(beta_fn) @ (rbk.EP2C(beta_bn).T)
    print(rbk.C2EP(dcm))

    print("# ------- CONCEPT CHECK 8 -------#")
    initial_set = (0.408248,0.,0.408248,0.816497)

    final_set = integrate_euler_parameter_set(initial_set)






