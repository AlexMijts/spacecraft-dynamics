import numpy as np
from spacecraftDynamics.control import control_estimation

# Handle both relative import (when run as module) and absolute import (when run directly)
try:
    from .. import ConceptCheck
except ImportError:
    from concept_checks import ConceptCheck

# Customize me
def get_initial_parameters():
    """Returns the common initial conditions """
    w_i = np.array([np.deg2rad(30), np.deg2rad(10), np.deg2rad(-20)])
    s_i = np.array([0.1, 0.2, -0.1])

    return w_i, s_i

def run_inertial_fixed_tracking(time, w_i, s_i, K, P, I, ftime, plotting):
    # Tracking reference attitude (Aligned with inertial frame)
    sigma_rn = [np.zeros(3)] * len(time)

    check = ConceptCheck("Concept Check 5: Feedback gain selection")
    results = check.run(
        control_estimation.propagate_inertial_tracking_control,
        time=time,
        reference_mrps=sigma_rn,
        I=I,
        gains=(K, P, None),
        sigma_bn_i=s_i,
        w_bn_i=w_i,
        show_plot=plotting,
    )
    check.print_result(
        label=f"Attitude mrp and angular rate error (inertial fixed tracking) at t = {ftime}s: "
    )

    return results

def main():
    plotting = True

    # --- Retrieve Input Data ---
    K = 5  # [Nm]
    I = np.array([[100, 0, 0], [0, 75, 0], [0, 0, 80]])
    w_i, s_i = get_initial_parameters()

    # --- Set Propagation Time ---
    h = 0.01
    ftime = 240  # [s]
    # Calculate time array once to pass into all functions
    time = np.linspace(0, ftime, int(ftime/h + 1))

    # --- Run Checks ---

    # Concept check 5 - Feedback gain selection
    damping_ratio = 1
    P = (
        damping_ratio
        * np.array([np.sqrt(K * I[0, 0]), np.sqrt(K * I[1, 1]), np.sqrt(K * I[2, 2])])
        * np.eye(3)
    )
    print(f"Critically damped diagonal P matrix :\n{P}")

    res = run_inertial_fixed_tracking(time, w_i, s_i, K, P, I, ftime, plotting)

    T = 2 * np.array([I[i, i] / P[i, i] for i in [0, 1, 2]])
    print(f"Decay times per axis T_i :\n{T}")

if __name__ == "__main__":
    main()

