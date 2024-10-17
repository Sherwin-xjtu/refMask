import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
# Generate some sample data
# np.random.seed(0)
# X = np.concatenate([
#     np.random.normal(0, 1, 200),
#     np.random.normal(5, 1, 200),
#     np.random.normal(10, 1, 200)
# ])
# X = X.reshape(-1, 1)

# # Fit a Gaussian Mixture Model with 3 components
# gmm = GaussianMixture(n_components=3, random_state=0)
# gmm.fit(X)

# # Plot the histogram of the data
# # plt.hist(X, bins=30, density=True, alpha=0.3, color='g')
# # Create a figure and axis
# # Create a figure and axis
# fig, ax = plt.subplots()

# # Plot the scatter plot of the data with some jitter in y-direction for better visibility
# jitter = 0.05 # you can adjust this value as needed
# ax.scatter(X, np.random.normal(0, jitter, len(X)), alpha=0.5, label='Data')

# # Generate an array of x values for the PDF plots
# x = np.linspace(X.min(), X.max(), 1000)

# # Plot the PDFs for each Gaussian component
# for i in range(gmm.n_components):
#     pdf = norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])) * gmm.weights_[i]
#     ax.plot(x, pdf, label=f'Component {i+1}')

# # Set the labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('Probability Density')
# ax.set_title('Gaussian Mixture Model and Data')
# ax.legend()

# # Display the plot
# # plt.show()


# fig, ax = plt.subplots()

# # Plot the histogram of the data
# ax.hist(X, bins=30, density=True, alpha=0.3, color='g', label='Data')

# # Generate an array of x values for the PDF plots
# x = np.linspace(X.min(), X.max(), 1000)

# # Plot the PDFs for each Gaussian component
# for i in range(gmm.n_components):
#     pdf = norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])) * gmm.weights_[i]
#     ax.plot(x, pdf, label=f'Component {i+1}')

# # Set the labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('Probability Density')
# ax.set_title('Gaussian Mixture Model and Data')
# ax.legend()

# Display the plot
# plt.show()


chrx_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/unique_q0q0region_counts.txt'
header_columns = ['chr', 'pos', 'r0']

chrX_df = pd.read_csv(chrx_file, sep='\t', names=header_columns)

data1 = chrX_df['r0']

chrX_df = chrX_df[chrX_df['pos'] < 45000]
plt.figure(figsize=(10, 8), dpi=300)
plt.scatter(chrX_df['pos'], chrX_df['r0'], alpha=1, s=1)
# plt.scatter(data2, data1, alpha=0.5)
plt.ylabel('r0')
plt.xlabel('POS')
# plt.title('Scatter plot of interval vs MAPQ (logarithmic scale)')
sns.despine(top=True, right=True)
plt.savefig('picture1.png')

X = chrX_df['r0'].tolist()
X = np.array(X).reshape(-1, 1)
# Fit a Gaussian Mixture Model with 3 components
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(X)

fig, ax = plt.subplots(dpi=300)

# Plot the histogram of the data
ax.hist(X, bins=30, density=True, alpha=0.3, color='g', label='Data')

# Generate an array of x values for the PDF plots
x = X

# Plot the PDFs for each Gaussian component
for i in range(gmm.n_components):
    pdf = norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])) * gmm.weights_[i]
    ax.plot(x, pdf, label=f'Component {i+1}')

# Set the labels and title
ax.set_xlabel('x')
ax.set_ylabel('Probability Density')
ax.set_title('Gaussian Mixture Model and Data')
ax.legend()
plt.savefig('picture2.png')