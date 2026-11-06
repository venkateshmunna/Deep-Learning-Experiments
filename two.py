import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Generate a correlated synthetic dataset
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.5, 100)
data = pd.DataFrame({'Feature_X': x, 'Feature_Y': y})

# Compute Covariance Matrix using Pandas
cov_matrix = data.cov()
print("Covariance Matrix:\n", cov_matrix)

# Compute Eigenvalues and Eigenvectors using NumPy
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix.values)
print("\nEigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Plot data points along with Eigenvectors (Principal Directions)
plt.figure(figsize=(8, 6))
plt.scatter(data['Feature_X'], data['Feature_Y'], alpha=0.7, label='Data points')

# Origin (Mean of the data)
mean_x, mean_y = data['Feature_X'].mean(), data['Feature_Y'].mean()

# Plot both eigenvectors scaled by their eigenvalues
for val, vec in zip(eigenvalues, eigenvectors.T):
    plt.quiver(mean_x, mean_y, vec[0] * np.sqrt(val), vec[1] * np.sqrt(val),
               angles='xy', scale_units='xy', scale=0.5, color='red', width=0.01,
               label=f'Eigenvector (Val: {val:.2f})')

plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Covariance and Eigenvectors Analysis')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
# Simple legend deduplication
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()