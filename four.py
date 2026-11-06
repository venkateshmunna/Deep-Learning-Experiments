import numpy as np
import matplotlib.pyplot as plt


# Define target probability distribution P and model distribution Q
P = np.array([0.8, 0.1, 0.1])
Q = np.array([0.6, 0.2, 0.2])

# Compute Shannon Entropy H(P)
shannon_entropy = -np.sum(P * np.log2(P))

# Compute Cross-Entropy H(P, Q)
cross_entropy = -np.sum(P * np.log2(Q))

# Compute Kullback-Leibler (KL) Divergence D_KL(P || Q)
kl_divergence = np.sum(P * np.log2(P / Q))

print(f"Probability Distribution P (Target): {P}")
print(f"Probability Distribution Q (Model):  {Q}")
print(f"--------------------------------------------")
print(f"Shannon Entropy H(P): {shannon_entropy:.4f} bits")
print(f"Cross-Entropy H(P, Q): {cross_entropy:.4f} bits")
print(f"KL Divergence D_KL(P || Q): {kl_divergence:.4f} bits")
print(f"Verification (H(P) + D_KL(P || Q)): {shannon_entropy + kl_divergence:.4f} bits (Should equal H(P,Q))")

# Let's plot the divergence comparison
metrics = ['Entropy H(P)', 'Cross-Entropy H(P,Q)', 'KL Divergence D_KL']
values = [shannon_entropy, cross_entropy, kl_divergence]

plt.figure(figsize=(8, 4))
plt.bar(metrics, values, color=['blue', 'green', 'orange'])
plt.ylabel('Information (bits)')
plt.title('Comparison of Information Theoretic Metrics')
plt.grid(axis='y', linestyle='--')
plt.show()