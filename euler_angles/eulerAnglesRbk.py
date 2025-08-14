import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk

if __name__ == "__main__":
    # 1
    e123 = np.deg2rad((10,20,30))
    C=rbk.euler3212C(e123)
    e313= rbk.C2Euler313(C)
    print(np.rad2deg(e313))
    # 2
    BN=rbk.euler3212C(np.deg2rad((10,20,30)))
    RN=rbk.euler3212C(np.deg2rad((-5,5,5)))
    BR=BN@RN.T
    print(np.rad2deg(rbk.C2Euler321(BR)))
