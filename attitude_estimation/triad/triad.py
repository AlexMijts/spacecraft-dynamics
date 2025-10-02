import numpy as np

def determine_triad_attitude(meas1_b : np.array, meas1_n: np.array, meas2_b : np.array, meas2_n: np.array, ) -> np.array:
    """
    Compute the [BN] dcm using the meas1 as a basis to create the intermediate T frame.

    Args:
        meas1_b (np.array): 3x1 measure 1 in body frame.
        meas1_n (np.array): 3x1 measure 1 in inertial frame.
        meas2_b (np.array): 3x1 measure 2 in body frame.
        meas2_n (np.array): 3x1 measure 2 in inertial frame.
    Returns:
        dcm (np.array): 3x3 BN matrix representing the attitude wrt inertial frame
    """

    # Compute the BT dcm using meas1
    t_1b=meas1_b
    cross=np.cross(meas1_b,meas2_b)
    norm_inv=1/np.linalg.norm(cross)
    t_2b=cross*norm_inv
    t_3b=np.cross(t_1b,t_2b)
    BT=np.column_stack((t_1b, t_2b, t_3b))

    # Compute the NT dcm using meas1
    t_1n=meas1_n
    cross=np.cross(meas1_n,meas2_n)
    norm_inv=1/np.linalg.norm(cross)
    t_2n=cross*norm_inv
    t_3n=np.cross(t_1n,t_2n)
    NT=np.column_stack((t_1n, t_2n, t_3n))

    return (BT @ NT.T)
