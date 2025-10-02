import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import pdb

from crp import add_crp

def integrate_crp_set(step_size : float,
                      angular_rates: np.ndarray,
                      initial_crp: np.ndarray) -> np.ndarray:
    '''
    Integrate the initial crp using the angular rates series with step_size [s] period between values
    '''

    crp_series = np.empty(len(angular_rates), dtype=np.ndarray)

    crp_series[0] = initial_crp

    for idx in range(1, len(angular_rates)):

        # Compute B matrix at this step
        q1 = crp_series[idx-1][0]
        q2 = crp_series[idx-1][1]
        q3 = crp_series[idx-1][2]
        B = np.array([[1+q1**2, q1*q2-q3, q1*q3 + q2],
                      [q2*q1 + q3, 1+q2**2, q2*q3 - q1],
                      [q3*q1 - q2, q2*q3 + q1, 1+q3**2]])
        q_increment = 0.5 * B @ angular_rates[idx] * step_size

        crp_series[idx] = crp_series[idx-1] + q_increment

    return crp_series

if __name__ == "__main__":

    print("# ------- CONCEPT CHECK 8 -------#")

    # Simulation setup
    init_step, final_step, step_duration = 0.0, 42.0, 0.0001
    nb_steps= int(np.round((final_step - init_step)/ step_duration))
    time = np.linspace(init_step, final_step, nb_steps+1)

    # Input
    initial_set = np.array([0.4, 0.2, -0.1])

    # Generate the discrete series of angular velocities
    omega_series = np.empty(len(time), dtype=np.ndarray)
    for idx, t in enumerate(time):
        omega_series[idx] = np.deg2rad(3) * np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])

    # Compute the successive values of beta using the angular velocities increments per steps
    final_set = integrate_crp_set(step_duration, omega_series, initial_set)

    # Plot result
    q1, q2, q3 = [], [], []
    for set in final_set:
        q1.append(set[0])
        q2.append(set[1])
        q3.append(set[2])

    norm = []
    for idx in range(0, len(q1)):
        norm.append(np.linalg.norm((q1[idx],q2[idx],q3[idx])))

    print("Final Norm value after " + str(time[nb_steps]) + "s : " + str(norm[-1]) + " (sampling period : "+ str(step_duration) + "s )")

    plt.figure()
    plt.plot(time, q1,  label='q1')
    plt.plot(time, q2,  label='q2')
    plt.plot(time, q3,  label='q3')
    plt.plot(time, norm,  label='NORM')
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True)
    plt.show()

