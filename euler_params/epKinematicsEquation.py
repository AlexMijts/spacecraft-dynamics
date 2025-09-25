import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import pdb

from Basilisk.utilities import RigidBodyKinematics as rbk

def integrate_euler_parameter_set(step_size : float,
                                  angular_rates: np.ndarray,
                                  initial_ep: np.ndarray) -> np.ndarray:
    '''
    Integrate the initial ep using the angular rates series with step_size [s] period between values
    '''

    ep_series = np.empty(len(angular_rates), dtype=np.ndarray)

    ep_series[0] = initial_ep

    for idx in range(1, len(angular_rates)):

        # Compute B matrix at this step
        beta0 = ep_series[idx-1][0]
        beta1 = ep_series[idx-1][1]
        beta2 = ep_series[idx-1][2]
        beta3 = ep_series[idx-1][3]
        B = np.array([[-beta1, -beta2, -beta3],
                      [beta0, -beta3, beta2],
                      [beta3, beta0, -beta1],
                      [-beta2, beta1, beta0]])

        ep_series[idx] = ep_series[idx-1] +  0.5 * B @ angular_rates[idx] * step_size

        # Normalise
        betanorm = np.linalg.norm(ep_series[idx])
        ep_series[idx] = (1/betanorm) * ep_series[idx]

    return ep_series

if __name__ == "__main__":

    print("# ------- CONCEPT CHECK 8 -------#")

    # Simulation setup
    init_step, final_step, step_duration = 0.0, 42.0, 0.0001
    nb_steps= int(np.round((final_step - init_step)/ step_duration))
    time = np.linspace(init_step, final_step, nb_steps+1)

    # Input
    initial_set = np.array([0.408248, 0.0, 0.408248, 0.816497])

    # Generate the discrete series of angular velocities
    omega_series = np.empty(len(time), dtype=np.ndarray)
    for idx, t in enumerate(time):
        omega_series[idx] = np.deg2rad(20) * np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])

    # Compute the successive values of beta using the angular velocities increments per steps
    final_set = integrate_euler_parameter_set(step_duration, omega_series, initial_set)

    # Plot result
    b0, b1, b2, b3 = [], [], [], []
    for set in final_set:
        b0.append(set[0])
        b1.append(set[1])
        b2.append(set[2])
        b3.append(set[3])

    norm = []
    for idx in range(0, len(b0)):
        norm.append(np.linalg.norm((b1[idx],b2[idx],b3[idx])))

    print("Final Norm value after " + str(time[nb_steps]) + "s : " + str(norm[-1]) + " (sampling period : "+ str(step_duration) + "s )")

    plt.figure()
    plt.plot(time, b0,  label='B0')
    plt.plot(time, b1,  label='B1')
    plt.plot(time, b2,  label='B2')
    plt.plot(time, b3,  label='B3')
    plt.plot(time, norm,  label='NORM')
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True)
    plt.show()

    # print("MRP EQUIVALENT for concept check 20 verification")

    # # Input
    # initial_set = rbk.MRP2EP(np.array([0.4, 0.2, -0.1]))

    # # Generate the discrete series of angular velocities
    # omega_series = np.empty(len(time), dtype=np.ndarray)
    # for idx, t in enumerate(time):
    #     omega_series[idx] = np.deg2rad(20) * np.array([np.sin(0.1*t), 0.01, np.cos(0.1*t)])

    # # Compute the successive values of beta using the angular velocities increments per steps
    # final_set = integrate_euler_parameter_set(step_duration, omega_series, initial_set)
    # for i in range(len(final_set)):
    #     final_set[i] = rbk.EP2MRP(final_set[i])

    # # Plot result
    # b0, b1, b2 = [], [], []
    # for set in final_set:
    #     b0.append(set[0])
    #     b1.append(set[1])
    #     b2.append(set[2])

    # norm = []
    # for idx in range(0, len(b0)):
    #     norm.append(np.linalg.norm((b0[idx], b1[idx],b2[idx])))

    # print("Final Norm value after " + str(time[nb_steps]) + "s : " + str(norm[-1]) + " (sampling period : "+ str(step_duration) + "s )")

    # plt.figure()
    # plt.plot(time, b0,  label='s0')
    # plt.plot(time, b1,  label='s1')
    # plt.plot(time, b2,  label='s2')
    # plt.plot(time, norm,  label='NORM')
    # plt.xlabel("Time [s]")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
