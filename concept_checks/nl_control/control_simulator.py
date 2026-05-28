import numpy as np
from spacecraftDynamics.control import control_estimation
from Basilisk.utilities import RigidBodyKinematics as rbk

# Handle both relative import (when run as module) and absolute import (when run directly)
try:
    from .. import ConceptCheck
except ImportError:
    from concept_checks import ConceptCheck

def compute_target_motion_mrp(t : float) -> tuple[np.ndarray, np.ndarray]:
    """Compute analytical sigma_rn and sigma_rn_dot"""

    f = 0.05  # [rad/s]
    s_rn = np.array([0.2 * np.sin(f * t), 0.3 * np.cos(f * t), -0.3 * np.sin(f * t)])
    s_rn_dot = f * np.array([0.2 * np.cos(f * t), -0.3 * np.sin(f * t), -0.3 * np.cos(f * t)])

    return s_rn, s_rn_dot

if __name__ == '__main__':
    # --- Input Data ---
    w_i = np.array([np.deg2rad(30), np.deg2rad(10), np.deg2rad(-20)])
    s_i = np.array([0.1, 0.2, -0.1])
    K = 5  # [Nm]
    P = 10 * np.eye(3)  # [Nms]
    I = np.array([[100, 0, 0], [0, 75, 0], [0, 0, 80]])

    # --- Concept Check 1: General 3-Axis Control ---

    # QUESTION 4

    # Propagation time
    h = 0.01
    ftime = 40 # [s]
    time = np.linspace(0, ftime, int(ftime/h + 1))

    # Tracking reference attitude (Aligned with inertial frame)
    sigma_rn = [np.zeros(3)] * len(time)

    check1 = ConceptCheck("Concept Check 1: General 3-Axis Control")
    check1.run(
        control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        I=I,
        K=K,
        P=P,
        sigma_bn_i=s_i,
        w_bn_i=w_i,
    )
    check1.print_result(label=f"Propagated attitudes (inertial fixed tracking)  at t = {ftime}s: ")

    # QUESTION 5

    # Propagation time
    h = 0.01
    ftime = 120 # [s]
    time = np.linspace(0, ftime, int(ftime/h + 1))

    # Tracking reference attitude (custom analytical tracking)
    sigma_rn = [np.zeros(3)] * len(time)
    sigma_rn_dot = [np.zeros(3)] * len(time)
    for i, t in enumerate(time):
        sigma_rn[i], sigma_rn_dot[i] = compute_target_motion_mrp(t)

    check2 = ConceptCheck("Concept Check 1: General 3-Axis Control")
    check2.run(
        control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        reference_mrp_rates=sigma_rn_dot,
        I=I,
        K=K,
        P=P,
        sigma_bn_i=s_i,
        w_bn_i=w_i,
    )
    check2.print_result(label=f"Propagated attitudes (custom analytical tracking) at t = {ftime} : ")
