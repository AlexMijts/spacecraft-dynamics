import numpy as np
from spacecraftDynamics.control import control_estimation
from Basilisk.utilities import RigidBodyKinematics as rbk

# Handle both relative import (when run as module) and absolute import (when run directly)
try:
    from .. import ConceptCheck
except ImportError:
    from concept_checks import ConceptCheck

if __name__ == '__main__':
    # --- Input Data ---
    w_i = np.array([np.deg2rad(30), np.deg2rad(10), np.deg2rad(-20)])
    s_i = np.array([0.1, 0.2, -0.1])
    K = 5  # [Nm]
    P = 10 * np.eye(3)  # [Nms]
    I = np.array([[100, 0, 0], [0, 75, 0], [0, 0, 80]])

    # Propagation time
    h = 0.01
    ftime = 41 # [s]
    time = np.linspace(0, ftime, int(ftime/h + 1))

    # Tracking reference attitude (Aligned with inertial frame)
    sigma_rn = [np.zeros(3)] * len(time)

    # --- Concept Check 1: General 3-Axis Control ---
    check1 = ConceptCheck("Concept Check 1: General 3-Axis Control")
    check1.run(control_estimation.propagate_inertial_tracking_control(time, sigma_rn, I, K, P, s_i, w_i))
    check1.print_result(label="Propagated attitudes : ")
