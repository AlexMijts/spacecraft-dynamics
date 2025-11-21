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
    check3.run(system.linear_momentum)
    check3.print_result(label="System linear momentum : ")


