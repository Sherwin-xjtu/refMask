# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create data
np.random.seed(0)
X = np.random.normal(0, 1, (100, 3))

# Create a 3D scatter plot
fig = plt.figure(figsize=(18, 6))

ax = fig.add_subplot(131, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.set_title('3D Scatter Plot')

# Create a 3D line plot
ax = fig.add_subplot(132, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
ax.set_title('3D Line Plot')

# Create a 3D surface plot
ax = fig.add_subplot(133, projection='3d')
X_surface = np.arange(-5, 5, 0.25)
Y_surface = np.arange(-5, 5, 0.25)
X_surface, Y_surface = np.meshgrid(X_surface, Y_surface)
Z_surface = np.sin(np.sqrt(X_surface**2 + Y_surface**2))
surf = ax.plot_surface(X_surface, Y_surface, Z_surface, cmap=plt.cm.coolwarm)
fig.colorbar(surf)
ax.set_title('3D Surface Plot')

plt.tight_layout()
plt.savefig('plot_test.png')
