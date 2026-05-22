import numpy as np
from ..attitude_coordinates.mrpKinematicsEquation import mrp_rate_to_angular_rate, angular_rate_to_mrp_rate
from ..attitude_coordinates.mrp import mrp_to_dcm

def propagate_inertial_tracking_control(
    time: np.ndarray,
    sigma_rn: np.ndarray,
    sigma_dot_rn = np.zeros(3),
    I=np.eye(3),
    K=1,
    P=np.eye(3),
    sigma_bn_i=np.array([0, 0, 0]),
    w_bn_i=np.array([0, 0, 0]),
    ext_torques=np.zeros(1),
) -> tuple:
    """
    Propagate free-tumbling spacecraft dynamics based on input control law, initial attitude and velocity

    Args:
        time: Timestamp array
        sigma_rn: Tracked inertial reference attitude MRP at each timestamp
        sigma_dot_rn (optional): Tracked inertial reference attitude MRP rate at each timestamp
        I (optional): S/C inertia matrix,
        K (optional): Proportional Gain
        P (optional): Angular rate error gain matrix
        sigma_bn_i (optional): Initial attitude MRP
        w_bn_i (optional): Initial angular velocities
        ext_torques (optional): perturbing torques at each timestamp
    Returns:
        tuple: The resulting error attitude and rate at the last timestamp (sigma_br, w_br_f)
    """
    # Check input consistency
    if len(sigma_rn) != len(time):
        raise ValueError("The input provided do not have consistent length")

    if len(ext_torques) > 1:
        if len(time) != len(ext_torques):
            raise ValueError("The input provided do not have consistent length")

    I_inv = np.linalg.inv(I)

    # Compute angular velocities increments at each timestamps
    sigma_bn = sigma_bn_i # BN MRP attitude buffer
    w_bn = w_bn_i  # BN rate buffer

    # Output tracking errors
    sigma_br = [np.zeros(3)] * len(time)
    w_br = [np.zeros(3)] * len(time)

    for i in range(1, len(time)):
        dt = float(time[i+1] - time[i])

        # --------------- Use Kinematics to compute current attitude and angular rate tracking errors  --------------- #

        # Attitude tracking error at t
        sigma_br[i] = sigma_bn - sigma_rn[i]

        # Reference attitude rate
        if len(sigma_dot_rn) != np.zeros(3):
            if len(sigma_dot_rn) != len(time):
                raise ValueError("Inconsistent sigma and sigma_dot series")
            # Use input-defined attitude rates
            s_rn_dot = sigma_dot_rn[i]
        else :
            # Derive the derivative tracked attitude using Newton method
            s_rn_dot = sigma_rn[i + 1] - sigma_rn[i]

        # Derive tracked angular rate from tracked inertial attitude rate (kinematic equation)
        w_rn = mrp_rate_to_angular_rate(s_rn_dot, sigma_rn)

        # Compute the current angular rate error
        w_br[i] = w_bn - w_rn

        # --------------- Propagate to next step using dynamic and defined control law  --------------- #

        # Control command computation
        if len(sigma_dot_rn) != np.zeros(3):
            w_rn_prev = mrp_rate_to_angular_rate(sigma_dot_rn[i-1], sigma_rn[i-1])
        else :
            w_rn_prev = mrp_rate_to_angular_rate(sigma_rn[i] - sigma_rn[i-1], sigma_rn[i-1])

        # Estimate w_rn_dot in body frame
        w_rn_dot = (w_rn - w_rn_prev)/dt

        # Map w_rn and w_rn_dot in body frame using sigma_br for DCM
        dcm_BR = mrp_to_dcm(sigma_br[i])
        w_rn_B = dcm_BR @ w_rn
        w_rn_dot_B = dcm_BR @ w_rn_dot

        # Compute control command (/!\ Map all vectorial elements to B frame)
        u = -K * sigma_br[i] - np.dot(P, w_br[i]) + np.dot(I, w_rn_dot_B - np.cross(w_bn, w_rn_B)) + np.cross(w_bn, np.dot(I, w_bn))

        # Derive the body angular acceleration from the dynamics
        w_bn_tilde = np.array([
            [0,        -w_bn[2],  w_bn[1]],
            [w_bn[2],  0,        -w_bn[0]],
            [-w_bn[1], w_bn[0],  0]
        ])

        w_bn_dot = I_inv @ (w_bn_tilde @ I @ w_bn) + u

        # Retreive the new w_bn and s_bn using simple Euler method
        s_bn = s_bn + angular_rate_to_mrp_rate(w_bn, s_bn) * dt # <-- Fixme
        w_bn = w_bn + w_bn_dot * dt


