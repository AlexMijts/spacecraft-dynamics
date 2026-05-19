import numpy as  np

def integrate_euler_set(angular_speed : float,
                      time_step: float,
                      yaw_pitch_roll: np.ndarray) -> np.ndarray:
    '''
        Assumes input in radians
    '''

    returned_set = np.zeros(3)
    t = time_step

    yaw = yaw_pitch_roll[0]
    pitch = yaw_pitch_roll[1]
    roll = yaw_pitch_roll[2]

    C = 1/np.cos(pitch) * np.array([[0, np.sin(roll), np.cos(roll)],
                                    [0, np.cos(roll)*np.cos(pitch), -np.sin(roll)*np.cos(pitch)],
                                    [np.cos(pitch), np.sin(roll)*np.sin(pitch), np.cos(roll)*np.sin(pitch)]])

    returned_set = C @ np.array([-10*np.cos(0.1*t)+np.deg2rad(40)+10,
                                 0.01*t+np.deg2rad(30),
                                 10*np.sin(0.1*t)+np.deg2rad(80)]) * angular_speed

    return returned_set

if __name__ == "__main__":

    #********************** Concept Check 9 **********************#

    # One minute integration
    w = np.deg2rad(20) #rad/sec

    # Angle set (yaw, pitch, roll)
    initial_set = np.deg2rad((40, 30, 80))

    integral_set = integrate_euler_set(w, 42, initial_set)

    print(np.rad2deg(integral_set))
    print(np.linalg.norm(integral_set))


