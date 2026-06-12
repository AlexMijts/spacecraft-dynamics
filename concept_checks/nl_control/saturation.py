import numpy as np
from spacecraftDynamics.control import control_estimation

# Handle both relative import (when run as module) and absolute import (when run directly)
try:
    from .. import ConceptCheck
except ImportError:
    from concept_checks import ConceptCheck

# Customize me
def get_initial_parameters():
    """Returns the common initial conditions"""

    w_i = np.array([np.deg2rad(30), np.deg2rad(10), np.deg2rad(-20)])
    s_i = np.array([0.1, 0.2, -0.1])

    return w_i, s_i


def compute_target_motion_mrp(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute analytical sigma_rn and sigma_rn_dot"""

    f = 0.05  # [rad/s]
    s_rn = np.array([0.2 * np.sin(f * t), 0.3 * np.cos(f * t), -0.3 * np.sin(f * t)])
    s_rn_dot = f * np.array([0.2 * np.cos(f * t), -0.3 * np.sin(f * t), -0.3 * np.cos(f * t)])

    return s_rn, s_rn_dot


def run_custom_analytical_tracking(time, w_i, s_i, K, P, u_max, I, ftime, plotting):
    # Tracking reference attitude (custom analytical tracking)
    sigma_rn = [np.zeros(3)] * len(time)
    sigma_rn_dot = [np.zeros(3)] * len(time)
    for i, t in enumerate(time):
        sigma_rn[i], sigma_rn_dot[i] = compute_target_motion_mrp(t)

    check = ConceptCheck("Concept Check 1: Saturated Control")
    check.run(
        control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        reference_mrp_rates=sigma_rn_dot,
        I=I,
        gains=(K, P, None),
        max_control=u_max,
        sigma_bn_i=s_i,
        w_bn_i=w_i,
        show_plot=plotting,
        tracking_error_time=60,
    )
    check.print_result(label=f"Attitude mrp and angular rate error (custom analytical saturated tracking) at t = {ftime}s: ")

def main():
    plotting = True

    # --- Retrieve Input Data ---
    w_i, s_i = get_initial_parameters()
    K = 5  # [Nm]
    K_i = 0.005 # [Nm]
    # K_i = 0 # [Nm]
    P = 10 * np.eye(3)  # [Nms]
    u_max = 1 # [Nm]
    I = np.array([[100, 0, 0], [0, 75, 0], [0, 0, 80]])

    # --- Set Propagation Time ---
    h = 0.01
    ftime = 240  # [s]
    # Calculate time array once to pass into all functions
    time = np.linspace(0, ftime, int(ftime/h + 1))

    # --- Run Checks ---
    run_custom_analytical_tracking(time, w_i, s_i, K, P, u_max, I, ftime, plotting)

if __name__ == "__main__":
    main()
