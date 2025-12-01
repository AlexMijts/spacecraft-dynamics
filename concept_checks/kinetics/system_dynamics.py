import numpy as np
from kinetics.system_properties import *
from kinetics.rigid_body import *
from attitude_coordinates.eulerAngles import compute_euler321_to_dcm
from concept_checks import ConceptCheck

if __name__ == '__main__':
    # --- Input Data ---
    system = System()

    p1 = Particle(np.array([1, -1, 2]), np.array([2, 1, 1]), 1)
    p2 = Particle(np.array([-1, -3, 2]), np.array([0, -1, 1]), 1)
    p3 = Particle(np.array([2, -1, -1]), np.array([3, 2, -1]), 2)
    p4 = Particle(np.array([3, -1, -2]), np.array([0, 0, 1]), 2)

    system = System(particles=[p1,p2,p3,p4])

    # --- Concept Check 2: kinetic Energy ---
    check2 = ConceptCheck("Concept Check 2: kinetic Energy")
    check2.run(system.kinetic_energy)
    check2.print_result(label="System energies [J] : ")

    # --- Concept Check 3: linear momentum ---
    check3 = ConceptCheck("Concept Check 3: linear momentum")
    check3.run(system.get_linear_momentum)
    check3.print_result(label="System linear momentum : ")

    # --- Concept Check 4 : angular momentum ---
    check4 = ConceptCheck("Concept Check : angular momentum")
    # angular momentum at cm :
    check4.run(system.get_angular_momentum)
    check4.print_result(label="System angular momentum : ")

    # angular momentum at origin :
    origin_angmom = system.compute_ang_mom(system.particles, (np.array([0,0,0]), np.array([0,0,0])))
    print("Angular momentum at origin : ", origin_angmom)

    # --- Concept Check 4 : angular momentum ---
    check4 = ConceptCheck("Concept Check : angular momentum")
    # angular momentum at cm :
    check4.run(system.get_angular_momentum)
    check4.print_result(label="System angular momentum : ")

    # --- Concept Check 5 : rigid body angular momentum ---
    check5 = ConceptCheck("Concept Check : rigid body angular momentum")
    Ic = np.array([[10, 1, -1], [1, 5, 1], [-1, 1, 8]])
    w_n = np.array([[0.01], [-0.01], [0.01]])
    w_b = compute_euler321_to_dcm(np.deg2rad(-10), np.deg2rad(10), np.deg2rad(5)) @ w_n
    check5.run(rbk_angular_momentum, Ic, w_b)
    check5.print_result(label="Rigid body angular momentum : ")

    check6 = ConceptCheck("Concept Check : Parallel axis theorem")
    bn = compute_euler321_to_dcm(np.deg2rad(-10), np.deg2rad(10), np.deg2rad(5))
    # Compute inertia in N frame
    I_n = bn.T @ Ic @ bn
    I_n_P = check6.run(recompute_inertia, I_n, 12.5, np.array([[-0.5],[0.5],[0.25]]))
    check6.print_result(label="New inertia at point P (N frame):")
    # Reset to B frame at the new point P
    I_b_P = bn @ I_n_P @ bn.T
    print("Inertia tensor at point P in Body frame B : \n", I_b_P)
