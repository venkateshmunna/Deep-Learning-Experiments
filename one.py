import numpy as np
import matplotlib.pyplot as plt
# Define vectors and matrices
u = np.array([2, 3])
v = np.array([4, -1])
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

# Perform fundamental operations
vector_add = u + v
dot_product = np.dot(u, v)
matrix_mul = np.dot(A, B)
hadamard_product = A * B
transpose_A = A.T

print(f"Vector Addition (u + v): {vector_add}")
print(f"Dot Product (u . v): {dot_product}")
print(f"Matrix Multiplication (A @ B):\n{matrix_mul}")
print(f"Hadamard (Element-wise) Product (A * B):\n{hadamard_product}")
print(f"Transpose of A:\n{transpose_A}")

# Visualization of vectors
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector u (2, 3)')
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector v (4, -1)')
plt.quiver(0, 0, vector_add[0], vector_add[1], angles='xy', scale_units='xy', scale=1, color='g', label='u + v')
plt.xlim(-1, 8)
plt.ylim(-2, 5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title("Vector Operations Visualization")
plt.show()