# Import necessary libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generate synthetic data with 3 features
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, n_features=3, random_state=0)

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=4)
gmm.fit(X)

# Extract the variances and create a DataFrame for visualization
variances = np.array([np.diag(cov_mat) for cov_mat in gmm.covariances_])
df_variances = pd.DataFrame(variances, columns=[f"Feature {i + 1}" for i in range(variances.shape[1])])

# Create a bar plot for each component
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    df_variances.loc[i].plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f"Component {i + 1}")
    ax.set_ylabel("Variance")

plt.tight_layout()



plt.savefig('plottest.png')

# plt.show()
