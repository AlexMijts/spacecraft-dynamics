import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import pdb

from Basilisk.utilities import RigidBodyKinematics as rbk

def angular_rate_to_mrp_rate(w, sigma) -> np.ndarray:
    '''
    Computes the corresponding mrp rate from the input angular rate and attitude
    '''
    s1 = sigma[0]
    s2 = sigma[1]
    s3 = sigma[2]
    norm_2 = np.dot(sigma, sigma)
    B = np.array([
        [1-norm_2+2*s1**2, 2*(s1*s2-s3), 2*(s1*s3+s2)],
        [2*(s2*s1+s3), 1-norm_2+2*s2**2, 2*(s2*s3-s1)],
        [2*(s3*s1-s2), 2*(s3*s2+s1), 1-norm_2+2*s3**2]
    ])

    return (0.25 * B @ w)

def mrp_rate_to_angular_rate(sigma_dot, sigma) -> np.ndarray:
    '''
    Computes the corresponding angular rate from the input mrp rate and attitude
    '''
    s1 = sigma[0]
    s2 = sigma[1]
    s3 = sigma[2]
    norm_2 = np.dot(sigma, sigma)
    B = np.array([
        [1-norm_2+2*s1**2, 2*(s1*s2-s3), 2*(s1*s3+s2)],
        [2*(s2*s1+s3), 1-norm_2+2*s2**2, 2*(s2*s3-s1)],
        [2*(s3*s1-s2), 2*(s3*s2+s1), 1-norm_2+2*s3**2]
    ])

    return ((4 / ((1 + norm_2) ** 2))) * B.T @ sigma_dot

def angular_rate_to_mrp_rate(w, sigma) -> np.ndarray:
    '''
        Computes the corresponding mrp rate from the input angular rate and attitude
    '''
    s1 = sigma[0]
    s2 = sigma[1]
    s3 = sigma[2]
    norm_2 = np.dot(sigma, sigma)
    B = np.array([
        [1-norm_2+2*s1**2, 2*(s1*s2-s3), 2*(s1*s3+s2)],
        [2*(s2*s1+s3), 1-norm_2+2*s2**2, 2*(s2*s3-s1)],
        [2*(s3*s1-s2), 2*(s3*s2+s1), 1-norm_2+2*s3**2]
    ])

    return (0.25 * B @ w)


def integrate_mrp_set(step_size : float,
                      propagation_time: float,
                      angular_rates: np.ndarray,
                      initial_mrp: np.ndarray,
                      show_plot = False) -> np.ndarray:
    '''
    Integrate the initial mrp using the angular rates series with step_size [s] period between values
    '''

    mrp_series = np.empty(len(angular_rates), dtype=np.ndarray)

    mrp_series[0] = initial_mrp

    for idx in range(1, len(angular_rates)):

        # Kinematics equation
        sigma_dot = angular_rate_to_mrp_rate(angular_rates[idx], mrp_series[idx - 1])

        # Euler integration
        mrp_series[idx] = mrp_series[idx - 1] + sigma_dot * step_size

        # Normalise to shadowset if norm exceeds 1
        if np.linalg.norm(mrp_series[idx]) > 1 :
            mrp_series[idx] = (-1/(np.dot(mrp_series[idx],mrp_series[idx]))) * mrp_series[idx]

    if show_plot:
        # Time
        nb_steps = int(np.round(propagation_time / step_size))
        time = np.linspace(0, propagation_time, nb_steps + 1)

        # Plot result
        s0, s1, s2 = [], [], []
        for set in mrp_series:
            s0.append(set[0])
            s1.append(set[1])
            s2.append(set[2])

        norm = []
        for idx in range(len(s0)):
            norm.append(np.linalg.norm((s0[idx], s1[idx], s2[idx])))

        print(
            "Final Norm value after "
            + str(time[nb_steps])
            + "s : "
            + str(norm[-1])
            + " (sampling period : "
            + str(step_size)
            + "s )"
        )

        plt.figure()
        plt.plot(time, s0, label="s0")
        plt.plot(time, s1, label="s1")
        plt.plot(time, s2, label="s2")
        plt.plot(time, norm, label="NORM")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.grid(True)
        plt.show()

    return mrp_series

if __name__ == "__main__":

    print("# ------- CONCEPT CHECK 20 -------#")

    # Simulation setup
    final_step, step_duration = 42.0, 0.0001
    nb_steps= int(np.round((final_step)/ step_duration))
    time = np.linspace(0, final_step, nb_steps+1)

    # Input
    initial_set = np.array([0.4, 0.2, -0.1])

    # Generate the discrete series of angular velocities
    omega_series = np.empty(len(time), dtype=np.ndarray)
    for idx, t in enumerate(time):
        omega_series[idx] = np.deg2rad(20) * np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])

    # Compute the successive values of beta using the angular velocities increments per steps
    final_set = integrate_mrp_set(step_duration, final_step, omega_series, initial_set, show_plot=True)
