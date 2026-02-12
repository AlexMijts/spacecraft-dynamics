import numpy as np
import control as ct
import matplotlib.pyplot as plt

# 1. Define the dynamics: dx/dt = [x2, -sin(x1)]
def pendulum_update(t, x, u, params):
    return np.array([x[1], -np.sin(x[0])])

# 2. Create the Nonlinear I/O System
# phase_plane_plot requires a NonlinearIOSystem object
sys = ct.nlsys(
    pendulum_update,
    states=2,
    inputs=0,
    name='pendulum'
)

# 3. Create the plot
plt.figure(figsize=(10, 8))

# We define the range of interest using 'pointdata' or 'gridspec'
# pointdata=[xmin, xmax, ymin, ymax]
ct.phase_plane_plot(
    sys,
    pointdata=[-2*np.pi, 2*np.pi, -3, 3],
    gridtype='meshgrid',
    gridspec=[20, 20],        # Density of the flow lines
    plot_streamlines=True,    # Draw the continuous flow lines
    plot_vectorfield=True,    # Draw the arrows
    plot_equilpoints=True,    # Automatically mark stable/unstable points
    plot_separatrices=True    # Draw the boundaries between different behaviors
)

# 4. Final Formatting
plt.title('Nonlinear Pendulum Phase Plane ($\ddot{x} + \sin(x) = 0$)')
plt.xlabel('Position $x$ [rad]')
plt.ylabel('Velocity $\dot{x}$ [rad/s]')
plt.axhline(0, color='black', alpha=0.2)
plt.axvline(0, color='black', alpha=0.2)
plt.show()
