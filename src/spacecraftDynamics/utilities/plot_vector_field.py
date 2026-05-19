import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# 1. Define your function: w = f(x, y, z)
def linear_transform(x, y, z):
    res = np.zeros(len(x))
    m1 = np.array(
        [
            [1.53947, -0.0422688, -0.190629],
            [-0.0422688, 1.4759, 0.459006],
            [-0.190629, 0.459006, 1.48463],
        ]
    )

    m2 = np.array(
        [
            [-0.984331, -1.10006, -0.478579],
            [-1.10006, 1.03255, 0.338318],
            [-0.478579, 0.338318, 1.45178],
        ]
    )

    m3 = np.array(
        [
            [-2.0353, 0.296916, -0.365128],
            [0.296916, -1.10369, -0.074481],
            [-0.365128, -0.074481, -2.86101],
        ]
    )

    for i, v in enumerate(zip(x,y,z)):
        res[i] = np.array([v[0], v[1], v[2]]).T @ m2 @ np.array([v[0], v[1], v[2]])
    return (res)

# 2. Generate 3D input data
# (Using random points is often better than a dense grid to avoid visual clutter)
num_points = 2000
x = np.random.uniform(-5, 5, num_points)
y = np.random.uniform(-5, 5, num_points)
z = np.random.uniform(-5, 5, num_points)

# 3. Calculate the scalar output
w = linear_transform(x, y, z)

# 4. Create the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the scatter, mapping the scalar 'w' to the 'c' (color) argument
# Using a colormap like 'plasma' or 'viridis' is recommended
scatter = ax.scatter(x, y, z, c=w, cmap='plasma', alpha=0.6, edgecolors='none')

# Add labels and a colorbar to explain the color mapping
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Vector to Scalar Visualization using Color Mapping')

# The colorbar is crucial for understanding the scalar values
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Scalar Output (w)')

plt.show()
