from concept_checks import ConceptCheck
import attitude_estimation.estimators as attest
from attitude_coordinates import prv

import numpy as np

def compare_dcms(dcm1: np.ndarray, dcm2: np.ndarray, label: str = "Delta Angle"):
    """
    Calculates and prints the principal rotation angle between two DCMs.
    Args:
        dcm1 (np.ndarray): The first direction cosine matrix.
        dcm2 (np.ndarray): The second direction cosine matrix.
        label (str): A label to print before the result.
    """
    delta_dcm = dcm1 @ dcm2.T
    angle, _ = prv.c2prv(delta_dcm)
    print(label)
    print(f"{np.rad2deg(angle):.6f} degrees")

if __name__ == '__main__':
    # --- Input Data ---
    v1b = np.array([0.8273, 0.5541, -0.092])
    v2b = np.array([-0.8285, 0.5522, -0.0955])
    v1n = np.array([-0.1517, -0.9669, 0.2050])
    v2n = np.array([-0.8393, 0.4494, -0.3044])

    # --- Concept Check 2: TRIAD Method ---
    triad_check = ConceptCheck("CONCEPT CHECK 2 - TRIAD")
    triad_dcm = triad_check.run(attest.determine_triad_attitude, v1b, v1n, v2b, v2n)
    triad_check.print_result(label="DCM Result:")

    # --- Delta Angle Calculation ---
    dcm1 = np.array([
        [0.969846, 0.17101, 0.173648],
        [-0.200706, 0.96461, 0.17101],
        [-0.138258, -0.200706, 0.969846]])
    dcm2 = np.array([
        [0.963592, 0.187303, 0.190809],
        [-0.223042, 0.956645, 0.187303],
        [-0.147454, -0.223042, 0.963592]])

    delta_angle_check = ConceptCheck("DELTA ANGLE")
    delta_angle_check.print_header()
    compare_dcms(dcm1, dcm2)

    # --- Concept Check 3 & 4: Davenport's q-Method ---
    devenport_check = ConceptCheck("CONCEPT CHECK 3, 4 - DAVENPORT Q-METHOD")
    weights = (1, 1)
    devenport_dcm = devenport_check.run(attest.devenport_q_method, (v1b, v2b), (v1n, v2n), weights)
    devenport_check.print_result(label="DCM Result:")

    # Compare Davenport's result with TRIAD
    compare_dcms(devenport_dcm, triad_dcm, label="Delta with TRIAD method:")
