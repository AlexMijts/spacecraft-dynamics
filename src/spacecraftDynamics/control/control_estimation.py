import numpy as np
import matplotlib.pyplot as plt

from ..attitude_coordinates.mrpKinematicsEquation import mrp_rate_to_angular_rate, angular_rate_to_mrp_rate
from ..attitude_coordinates.mrp import mrp_to_dcm, mrp_subtraction


def compute_linear_rate(x_prev, x_now, dt):
    """Compute first order rate approximation"""
    return (x_now - x_prev) / dt


def generate_control(K, P, I, w_ref, w_dot_ref, w_bn, sigma_br, dt) -> np.ndarray:
    """Compute control law"""
    # Map w_rn and w_rn_dot in body frame using sigma_br for DCM
    dcm_BR = mrp_to_dcm(sigma_br)
    w_rn_B = dcm_BR @ w_ref
    w_br = w_bn - w_rn_B
    w_rn_dot_B = dcm_BR @ w_dot_ref

    # Compute control command
    u = -K * sigma_br - np.dot(P, w_br) + np.dot(I, w_rn_dot_B - np.cross(w_bn, w_rn_B)) + np.cross(w_bn, np.dot(I, w_bn))

    return u

def propagate_inertial_tracking_control(
    time: np.ndarray,
    reference_mrps: np.ndarray,
    I: np.ndarray,
    reference_mrp_rates=None,
    K=1,
    P=np.eye(3),
    sigma_bn_i=np.array([0, 0, 0]),
    w_bn_i=np.array([0, 0, 0]),
    ext_torques=np.zeros(1),
    show_plot=True,
) -> tuple[float, float]:

    N = len(time)

    # Check input consistency
    if len(reference_mrps) != N:
        raise ValueError("The input provided do not have consistent length")

    if len(ext_torques) > 1 and len(ext_torques) != N:
        raise ValueError("The input provided do not have consistent length")

    I_inv = np.linalg.inv(I)

    # Initialize buffers
    sigma_bn = np.array(sigma_bn_i, dtype=float)
    w_bn = np.array(w_bn_i, dtype=float)

    # Pre-allocate output tracking errors to exactly match the length of 'time'
    sigma_br = np.zeros((N, 3))
    w_br = np.zeros((N, 3))

    # Pre-allocate solution series
    w_bn_series = np.zeros((N, 3))
    w_rn_series = np.zeros((N, 3))
    sigma_bn_series = np.zeros((N, 3))

    # Store initial conditions at t=0
    sigma_bn_series[0] = sigma_bn
    w_bn_series[0] = w_bn

    # Loop runs up to N-2 to allow forward-looking finite difference at i+1
    for i in range(N - 1):
        dt = float(time[i+1] - time[i])
        s_rn = reference_mrps[i]

        # --------------- Compute current attitude and angular rate tracking errors  --------------- #

        # Correct MRP error composition
        sigma_br[i] = mrp_subtraction(sigma_bn, reference_mrps[i])

        if reference_mrp_rates is not None:
            if len(reference_mrp_rates) != N:
                raise ValueError("Inconsistent sigma and sigma_dot series")
            s_rn_dot = reference_mrp_rates[i]

            # Handle the i=0 edge case
            prev_idx = max(0, i - 1)
            w_rn_prev = mrp_rate_to_angular_rate(reference_mrp_rates[prev_idx], reference_mrps[prev_idx])
        else:
            # Linear numerical derivative
            s_rn_dot = (reference_mrps[i + 1] - reference_mrps[i]) / dt

            # Handle the i=0 edge case
            prev_idx = max(0, i - 1)
            mrp_rate = compute_linear_rate(reference_mrps[i], reference_mrps[prev_idx], dt)
            w_rn_prev = mrp_rate_to_angular_rate(mrp_rate, reference_mrps[prev_idx])

        w_rn = mrp_rate_to_angular_rate(s_rn_dot, s_rn)

        # Prevent dividing by zero on the very first loop iteration for w_rn_dot
        if i == 0:
            w_rn_dot = np.zeros(3)
        else:
            w_rn_dot = compute_linear_rate(w_rn, w_rn_prev, dt)

        # Store w_rn and compute rate error
        w_rn_series[i] = w_rn
        w_br[i] = w_bn - w_rn

        # --------------- Compute the control to apply at this step  --------------- #

        u = generate_control(K, P, I, w_rn, w_rn_dot, w_bn, sigma_br[i], dt)

        # --------------- Propagate to next step using dynamics and defined control law  --------------- #

        w_bn_tilde = np.array([
            [0, -w_bn[2],  w_bn[1]],
            [w_bn[2], 0, -w_bn[0]],
            [-w_bn[1], w_bn[0], 0]
        ])

        # Corrected sign on the gyroscopic torque term
        w_bn_dot = I_inv @ (u - (w_bn_tilde @ I @ w_bn))

        # Retreive the new w_bn and s_bn using Euler method
        sigma_bn_next = sigma_bn + angular_rate_to_mrp_rate(w_bn, sigma_bn) * dt
        w_bn_next = w_bn + w_bn_dot * dt

        # Normalise to shadowset if norm exceeds 1
        if np.linalg.norm(sigma_bn_next) > 1:
            sigma_bn_next = (-1 / (np.dot(sigma_bn_next, sigma_bn_next))) * sigma_bn_next

        # Store the propagated state at step i+1
        sigma_bn_series[i+1] = sigma_bn_next
        w_bn_series[i+1] = w_bn_next

        # Update the state buffers for the next iteration
        sigma_bn = sigma_bn_next
        w_bn = w_bn_next

    # Calculate final errors for the return statement based on the final states
    sigma_br[-1] = mrp_subtraction(sigma_bn_series[-1], reference_mrps[-1])
    # w_br calculation for the final index would require w_rn[-1],
    # assuming w_rn settles, returning index -2 is safe.

    if show_plot:
        w_bn_series = np.array(w_bn_series)
        w_rn_series = np.array(w_rn_series)
        reference_mrps = np.array(reference_mrps)
        sigma_bn_series = np.array(sigma_bn_series)

        mrp_norm = [np.dot(i, i) for i in sigma_bn_series]

        plt.figure(0)
        plt.plot(time, sigma_bn_series[:, 0], 'g')
        plt.plot(time, reference_mrps[:, 0], 'g--')
        plt.plot(time, sigma_bn_series[:, 1], 'b')
        plt.plot(time, reference_mrps[:, 1], 'b--')
        plt.plot(time, sigma_bn_series[:, 2], 'r')
        plt.plot(time, reference_mrps[:, 2], 'r--')
        plt.plot(time, mrp_norm, 'k')
        plt.title('Attitude (sigma) series')
        plt.legend(["sigma_1", "sigma_1_target", "sigma_2", "sigma_2_target", "sigma_3", "sigma_3_target", "norm^2"])
        plt.grid()
        plt.figure(1)
        plt.plot(time, w_bn_series[:, 0], "b")
        plt.plot(time, w_rn_series[:, 0], "b--")
        plt.plot(time, w_bn_series[:, 1], "r")
        plt.plot(time, w_rn_series[:, 1], "r--")
        plt.plot(time, w_bn_series[:, 2], "g")
        plt.plot(time, w_rn_series[:, 2], "g--")
        plt.title("Rate (w) series")
        plt.legend(["w_1", "w_1_target", "w_2", "w_2_target", "w_3", "w_3_target"])
        plt.grid()
        plt.show()

    return float(np.linalg.norm(sigma_br[-2])), float(np.linalg.norm(w_br[-2]))
