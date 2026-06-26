import numpy as np
from spacecraftDynamics.control import nl_control_estimation

# Handle both relative import (when run as module) and absolute import (when run directly)
try:
    from .. import ConceptCheck
except ImportError:
    from concept_checks import ConceptCheck

# Customize me
def get_initial_parameters():
    """Returns the common initial conditions """
    w_i = np.array([np.deg2rad(3), np.deg2rad(1), np.deg2rad(-2)])
    s_i = np.array([0.1, 0.2, -0.1])

    return w_i, s_i

def compute_target_motion_mrp(t : float) -> tuple[np.ndarray, np.ndarray]:
    """Compute analytical sigma_rn and sigma_rn_dot"""

    f = 0.05  # [rad/s]
    s_rn = np.array([0.2 * np.sin(f * t), 0.3 * np.cos(f * t), -0.3 * np.sin(f * t)])
    s_rn_dot = f * np.array([0.2 * np.cos(f * t), -0.3 * np.sin(f * t), -0.3 * np.cos(f * t)])

    return s_rn, s_rn_dot

def run_inertial_fixed_tracking(time, w_i, s_i, K, P, I, ftime, plotting):
    # Tracking reference attitude (Aligned with inertial frame)
    sigma_rn = [np.zeros(3)] * len(time)

    check = ConceptCheck("Concept Check 1: General 3-Axis Control")
    check.run(
        nl_control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        I=I,
        gains=(K, P, None),
        sigma_bn_i=s_i,
        w_bn_i=w_i,
        show_plot=plotting
    )
    check.print_result(label=f"Attitude mrp and angular rate error (inertial fixed tracking) at t = {ftime}s: ")

def run_custom_analytical_tracking(time, w_i, s_i, K, P, I, ftime, plotting):
    # Tracking reference attitude (custom analytical tracking)
    sigma_rn = [np.zeros(3)] * len(time)
    sigma_rn_dot = [np.zeros(3)] * len(time)
    for i, t in enumerate(time):
        sigma_rn[i], sigma_rn_dot[i] = compute_target_motion_mrp(t)

    check = ConceptCheck("Concept Check 1: General 3-Axis Control")
    check.run(
        nl_control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        reference_mrp_rates=sigma_rn_dot,
        I=I,
        gains=(K, P, None),
        sigma_bn_i=s_i,
        w_bn_i=w_i,
        show_plot=plotting,
    )
    check.print_result(label=f"Attitude mrp and angular rate error (custom analytical tracking) at t = {ftime}s: ")

def run_custom_tracking_with_torque(time, w_i, s_i, K, P, I, ftime, plotting):
    """Runs Concept Check 2: Custom analytical tracking with a known external torque."""
    # Tracking reference attitude (custom analytical tracking)
    sigma_rn = [np.zeros(3)] * len(time)
    sigma_rn_dot = [np.zeros(3)] * len(time)
    for i, t in enumerate(time):
        sigma_rn[i], sigma_rn_dot[i] = compute_target_motion_mrp(t)

    L = [np.array([0.5, -0.3, 0.2])] * len(time)  # [Nm]

    check = ConceptCheck("Concept Check 2: Asymptotic stability")
    check.run(
        nl_control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        reference_mrp_rates=sigma_rn_dot,
        I=I,
        gains=(K, P, None),
        sigma_bn_i=s_i,
        w_bn_i=w_i,
        ext_torques=(L, None),
        show_plot=plotting
    )
    check.print_result(label=f"Attitude mrp and angular rate error (custom analytical tracking with external torque) at t = {ftime}s: ")


def run_custom_tracking_unmodelled_torque_control(time, w_i, s_i, K, P, K_i, I, ftime, plotting):
    """Runs Concept Check 4: Custom analytical tracking with a unknown external torque and 'pid' style control"""
    # Tracking reference attitude (custom analytical tracking)
    sigma_rn = [np.zeros(3)] * len(time)
    sigma_rn_dot = [np.zeros(3)] * len(time)
    for i, t in enumerate(time):
        sigma_rn[i], sigma_rn_dot[i] = compute_target_motion_mrp(t)

    L = [np.array([0.5, -0.3, 0.2])] * len(time)  # [Nm]

    check = ConceptCheck("Concept Check 4: Asymptotic stability")
    check.run(
        nl_control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        reference_mrp_rates=sigma_rn_dot,
        I=I,
        gains=(K, P, K_i),
        sigma_bn_i=s_i,
        w_bn_i=w_i,
        ext_torques=(None, L),
        show_plot=plotting,
    )
    check.print_result(label=f"Attitude mrp and angular rate error (custom analytical tracking with external torque) at t = {ftime}s: ")

def main():
    plotting = True

    # --- Retrieve Input Data ---
    w_i, s_i = get_initial_parameters()
    K = 5  # [Nm]
    K_i = 0.005 # [Nm]
    # K_i = 0 # [Nm]
    P = 10 * np.eye(3)  # [Nms]
    I = np.array([[100, 0, 0], [0, 75, 0], [0, 0, 80]])

    # --- Set Propagation Time ---
    h = 0.01
    ftime = 240  # [s]
    # Calculate time array once to pass into all functions
    time = np.linspace(0, ftime, int(ftime/h + 1))

    # --- Run Checks ---
    # run_inertial_fixed_tracking(time, w_i, s_i, K, P, I, ftime, plotting)
    # run_custom_analytical_tracking(time, w_i, s_i, K, P, I, ftime, plotting)
    # run_custom_tracking_with_torque(time, w_i, s_i, K, P, I, ftime, plotting)
    run_custom_tracking_unmodelled_torque_control(time, w_i, s_i, K, P, K_i, I, ftime, plotting)
    print(f'Steady state error z : {1/ K_i * np.linalg.inv(P) @ np.array([0.5, -0.3, 0.2])}')

if __name__ == "__main__":
    main()
