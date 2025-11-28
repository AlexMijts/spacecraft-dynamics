import numpy as np
from kinetics.system_properties import *
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



