import numpy as np
from nl_control import control_estimation
from Basilisk.utilities import RigidBodyKinematics as rbk
from concept_checks import ConceptCheck

if __name__ == '__main__':
    # --- Input Data ---

    w_i = np.array([np.deg2rad(30), np.deg2rad(10), np.deg2rad(-20)])
    s_i = np.array([0.1, 0.2, -0.1])

    # --- Concept Check 1: General 3-Axis Control ---
    check1 = ConceptCheck("Concept Check 1: General 3-Axis Control")
    check1.run(control_estimation.inertial_attitude_control_propagator(120.0, s_i, w_i))
    check1.print_result(label="Propagated attitudes : ")
