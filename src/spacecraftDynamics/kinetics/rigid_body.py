import numpy as np
from ..attitude_coordinates.common.matrixOperations import skew_symmetric

def rbk_angular_momentum(inertia : np.ndarray, ang_vel : np.ndarray) -> np.ndarray :
    """Compute angular momentum at center of mass based on inertia matrix and angular velocity"""
    ang_vel.reshape(3, 1)
    return (inertia @ ang_vel)

def recompute_inertia(inertia : np.ndarray, mass:float, r: np.ndarray) -> np.ndarray:
    """Computes the inertia tensor at any point r (position relative to CM) using input inertia tensor and mass"""
    rc_tilde = skew_symmetric(r)
    return(inertia + mass * rc_tilde @ rc_tilde.T)

