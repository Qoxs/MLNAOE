import numpy as np
import matplotlib.pyplot as plt

# Generate the objective function with noise
def generate_objective_func(n_sample, noise_std):
    x = np.linspace(0, 6 * np.pi, n_sample)
    normal_noise = np.random.normal(0, noise_std, n_sample)
    y = np.sin(x) + x / 2 + normal_noise
    return x, y

# Fit a polynomial to the data
def fit_polynomial(x, y, degree):
    return np.polyfit(x, y, degree)

# Calculate Mean Squared Error
def MSE_Error(y_true, y_predict):
    return np.mean((y_true - y_predict) ** 2)

# Parameters
n_repeat = 100
n_sample = 100
ratio = 0.8
noise_std = 0.5

n_train = int(n_sample * ratio)
degrees = range(1, 20)

# Initialize lists to store results
all_biases = []
all_variances = []
all_total_errors = []

# Repeat experiments
for _ in range(n_repeat):
    x, y = generate_objective_func(n_sample, noise_std)
    indices = np.random.permutation(n_sample)
    x_train, x_test = x[indices[:n_train]], x[indices[n_train:]]
    y_train, y_test = y[indices[:n_train]], y[indices[n_train:]]

    biases = []
    variances = []
    total_error = []

    for degree in degrees:
        model = fit_polynomial(x_train, y_train, degree)
        y_test_pred = np.polyval(model, x_test)

        # True function without noise
        y_true = np.sin(x_test) + x_test / 2

        # Bias and variance calculations
        bias_squared = np.mean((y_true - y_test_pred) ** 2)
        variance = np.var(y_test_pred)

        biases.append(bias_squared)
        variances.append(variance)
        total_error.append(bias_squared + variance + noise_std**2)

    all_biases.append(biases)
    all_variances.append(variances)
    all_total_errors.append(total_error)

# Calculate averages over all repetitions
avg_biases = np.mean(all_biases, axis=0)
avg_variances = np.mean(all_variances, axis=0)
avg_total_errors = np.mean(all_total_errors, axis=0)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(degrees, avg_biases, label='Bias^2')
plt.plot(degrees, avg_variances, label='Variance')
plt.plot(degrees, avg_total_errors, label='Total Error')
plt.plot(degrees, [noise_std**2] * len(degrees), label='Noise', linestyle='--')

plt.xlabel('Degree')
plt.title('Bias-Variance Trade-off')
plt.legend()
plt.grid(True)
plt.show()
