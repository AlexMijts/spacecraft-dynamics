import numpy as np
from ..attitude_coordinates.mrpKinematicsEquation import integrate_mrp_set

def inertial_attitude_control_propagator(
    duration: float,  # Propagation time in seconds
    sigma_i=np.array([0.1, 0.2, -0.1]),  # Initial attitude MRP
    w_i=np.array([0, 0, 0]),  # Initial angular velocities
) -> np.ndarray:
    # Constants
    I = np.array([[100, 0, 0], [0, 75, 0], [0, 0, 80]])
    I_inv = np.linalg.inv(I)
    # Control params
    P = 10 * np.eye(3)
    K = 5
    # Simulation setup
    step_duration = 0.001
    nb_steps= int(np.round(duration/ step_duration)) + 1
    w_out = [np.zeros(3)] * nb_steps

    # Compute angular velocities increments at each timestamps
    sigma_prev = sigma_i
    w_prev = w_i
    w_dot = 0

    for idx in range(1, len(w_out)):
        w_dot = I_inv @ (-K * sigma_prev - P @ w_prev)
        w_out[idx] = w_prev + w_dot
        w_prev = w_out[idx]

    # Retreive the corresponding attitude using kinematics equation
    sigmas = integrate_mrp_set(step_duration, duration, w_out, sigma_i, show_plot=True)

    return sigmas
