import numpy as np
from attitude_coordinates.ep import ep2c

def determine_triad_attitude(meas1_b : np.array, meas1_n: np.array, meas2_b : np.array, meas2_n: np.array, ) -> np.array:
    """
    Compute the [BN] dcm using the triad method based on meas_1 direction

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

def devenport_q_method(v_b_list, v_n_list, w_list):
    """
    Compute the [BN] dcm using the devenport q method.

    Args:
        v_b_list (tuple): tuple of 3x1 measurements vectors in B S/C frame
        v_n_list (tuple): tuple of corresponding 3x1 inertial vectors in N frame
        w_list (tuple): tuple of scalar weights associated with each measurement.
    Returns:
        dcm (np.array): 3x3 BN matrix representing the attitude wrt inertial frame
    """
    if len(v_b_list) != len(v_n_list) or len(v_b_list) != len(w_list):
        raise(RuntimeError("the list of measurements-pair and weights need to be of identical length"))

    B = np.zeros((3,3))

    # Ensure all vectors are 3x1 column arrays
    v_b_list = [np.asarray(vb).reshape(3, 1) for vb in v_b_list]
    v_n_list = [np.asarray(vn).reshape(3, 1) for vn in v_n_list]

    # Compute B
    for vb, vn, w in zip(v_b_list, v_n_list, w_list):
        # # Normalize inputs
        vb = vb/np.linalg.norm(vb)
        vn = vn/np.linalg.norm(vn)
        # Compute B
        B += w * vb @ vn.T

    # Compute S, sigma,Z and create K matrix
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        [B[1,2] - B[2,1]],
        [B[2,0] - B[0,2]],
        [B[0,1] - B[1,0]]
    ])
    K = np.array([sigma])
    K = np.concatenate((K, Z.T), axis=None)
    K= np.vstack([K, np.concatenate((Z, S-sigma*np.eye(3)), axis=1)])

    # Compute eigen value-vector pairs
    val, vec = np.linalg.eig(K)
    # Extract the biggest eigen value and its corresponding vector
    index_max = max(range(len(val)), key=val.__getitem__)
    eig_pair = (val[index_max], vec[:,index_max])

    return ep2c(eig_pair[1])

def newton_raphson_lambda_iterator(initial_lambda : float, mat : np.ndarray, epsilon : float):
    """
    Returns the lambda value once the difference between two iterations goes below the input threshold using Newton-Raphson method

    Args:
        initial_lambda (float): initial guess of the lambda value
        mat (np.ndarray): matrix used to solve eigen-vector/value equation
        epsilon (float): lambda difference threshold stopping the iterations
    Returns:
        optimal_lambda (float): converged lambda value
    """
    cond = bool(True)
    l_i = initial_lambda
    while cond:
        # Compute f(l_i)/f'(l_i) using jacobi's formula for f(l) = det(mat-l*I)
        ratio = 1/np.trace(np.linalg.inv(mat - l_i * np.eye(len(mat))) @ (- np.eye(len(mat))) )
        l_f = l_i - ratio

        # Check if more iterations are needed
        cond = (np.abs(l_i - l_f) > epsilon)
        if cond:
            l_i = l_f

    return l_f

def quest_method(v_b_list, v_n_list, w_list):
    """
    Compute the [BN] dcm using the QUEST method

    Args:
        v_b_list (tuple): tuple of 3x1 measurements vectors in B S/C frame
        v_n_list (tuple): tuple of corresponding 3x1 inertial vectors in N frame
        w_list (tuple): tuple of scalar weights associated with each measurement.
    Returns:
        b (np.array): 4x1 euler param representing the attitude wrt inertial frame
    """
    if len(v_b_list) != len(v_n_list) or len(v_b_list) != len(w_list):
        raise(RuntimeError("the list of measurements-pair and weights need to be of identical length"))

    B = np.zeros((3,3))

    # Ensure all vectors are 3x1 column arrays
    v_b_list = [np.asarray(vb).reshape(3, 1) for vb in v_b_list]
    v_n_list = [np.asarray(vn).reshape(3, 1) for vn in v_n_list]

    # Compute B
    for vb, vn, w in zip(v_b_list, v_n_list, w_list):
        # # Normalize inputs
        vb = vb/np.linalg.norm(vb)
        vn = vn/np.linalg.norm(vn)
        # Compute B
        B += w * vb @ vn.T

    # Compute S, sigma,Z and create K matrix
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        [B[1,2] - B[2,1]],
        [B[2,0] - B[0,2]],
        [B[0,1] - B[1,0]]
    ])
    K = np.array([sigma])
    K = np.concatenate((K, Z.T), axis=None)
    K= np.vstack([K, np.concatenate((Z, S-sigma*np.eye(3)), axis=1)])

    # Find the optimal lambda value from Newton-Raphson method
    l_o = newton_raphson_lambda_iterator(sum(w_list), K, 1e-15)

    # Compute the corresponding CRP
    q = np.linalg.inv( (l_o + sigma) * np.eye(3) - S ) @ Z
    q = np.asarray(q).reshape(3, 1)
    q_factor = np.vstack((np.array([[1.0]]), q))
    # Derive the corresponding euler parameter

    b = ( 1/( 1 + q.T @ q) ) * q_factor
    b = b/np.linalg.norm(b)

    return b.flatten()


