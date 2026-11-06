import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("=== Experiment 5: Bias-Variance, Overfitting and Underfitting ===")


# Generate ground truth non-linear function with noise
def true_fun(X):
    return np.sin(1.5 * np.pi * X)


np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.15

X_test = np.linspace(0, 1, 100)

# Fit 3 polynomial models: Degree 1 (Underfit / High Bias), Degree 4 (Good Fit), Degree 15 (Overfit / High Variance)
degrees = [1, 4, 15]
plt.figure(figsize=(15, 5))

for idx, degree in enumerate(degrees):
    ax = plt.subplot(1, len(degrees), idx + 1)

    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models
    y_pred_train = pipeline.predict(X[:, np.newaxis])
    train_mse = mean_squared_error(y, y_pred_train)

    y_test_pred = pipeline.predict(X_test[:, np.newaxis])
    test_mse = mean_squared_error(true_fun(X_test), y_test_pred)

    # Plotting
    plt.plot(X_test, y_test_pred, label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function", linestyle='--')
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(f"Degree {degree}\nTrain MSE: {train_mse:.2e}\nTest MSE: {test_mse:.2e}")

plt.tight_layout()
plt.show()