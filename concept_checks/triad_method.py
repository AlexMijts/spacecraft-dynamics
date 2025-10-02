import attitude_estimation.triad.triad as attest
import attitude_coordinates.prv.prv as prv
import numpy as np

if __name__ == '__main__':
    v1b = np.array([0.8273, 0.5541, -0.092])
    v2b = np.array([-0.8285, 0.5522, -0.0955])
    v1n = np.array([-0.1517, -0.9669, 0.2050])
    v2n = np.array([-0.8393, 0.4494, -0.3044])

    dcm = attest.determine_triad_attitude(v1b, v1n, v2b, v2n)
    print("# ------- CONCEPT CHECK 2 -------#")
    print(dcm)
    print("delta angle")
    dcm1 = np.array([
        [0.969846, 0.17101, 0.173648],
        [-0.200706, 0.96461, 0.17101],
        [-0.138258, -0.200706, 0.969846]])
    dcm2 = np.array([
        [0.963592, 0.187303, 0.190809],
        [-0.223042, 0.956645, 0.187303],
        [-0.147454, -0.223042, 0.963592]])
    angle, _ = prv.c2prv(dcm1@dcm2.T)
    print(np.rad2deg(angle))



