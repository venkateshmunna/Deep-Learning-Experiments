from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


print("=== Experiment 3: Probability Distributions and Bayes' Theorem ===")

# 1. Plotting Normal and Binomial Distributions
x_continuous = np.linspace(-4, 4, 200)
normal_pdf = stats.norm.pdf(x_continuous, loc=0, scale=1)

k_discrete = np.arange(0, 20)
binomial_pmf = stats.binom.pmf(k_discrete, n=20, p=0.5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x_continuous, normal_pdf, 'r-', lw=2, label='Normal PDF')
axes[0].set_title('Normal Distribution (Continuous)')
axes[0].grid(True)
axes[0].legend()

axes[1].stem(k_discrete, binomial_pmf, linefmt='b-', markerfmt='bo', basefmt='r-', label='Binomial PMF')
axes[1].set_title('Binomial Distribution (Discrete: n=20, p=0.5)')
axes[1].grid(True)
axes[1].legend()
plt.show()

# 2. Demonstration of Bayes' Theorem
# Problem: P(Disease | Positive Test)
# Given: P(Disease) = 0.01, P(Positive | Disease) (Sensitivity) = 0.99, P(Positive | No Disease) (False Positive) = 0.05
p_disease = 0.01
p_no_disease = 1 - p_disease
p_pos_given_disease = 0.99
p_pos_given_no_disease = 0.05

# Bayes' Theorem formulation:
# P(Disease | Positive) = (P(Positive | Disease) * P(Disease)) / P(Positive)
p_positive = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * p_no_disease)
p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive

print(f"Prior Probability of Disease P(D): {p_disease}")
print(f"Marginal Probability of Positive Test P(Pos): {p_positive:.4f}")
print(f"Posterior Probability P(D | Pos) via Bayes' Theorem: {p_disease_given_positive:.4f}")