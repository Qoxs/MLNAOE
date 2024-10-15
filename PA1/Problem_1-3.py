import numpy as np
import matplotlib.pyplot as plt

def generate_objective_func(n_sample, noise_std):
    x = np.linspace(0, 6 * np.pi, n_sample)
    normal_noise = np.random.normal(0, noise_std, n_sample)
    y = np.sin(x) + x / 2 + normal_noise
    return x, y

def fit_polynomial(x, y, degree):
    return np.polyfit(x, y, degree)

def MSE_Error(y_true, y_predict):
    return np.mean((y_true - y_predict) ** 2)

# Parameters
n_sample = 250
noise_std = 0.5

# Generate data
x, y_noisy = generate_objective_func(n_sample, noise_std)

# Calculate true function values
y_true = np.sin(x) + x / 2

# Fit polynomials of different degrees
degrees = range(1, 20)
errors = []

for degree in degrees:
    model = fit_polynomial(x, y_noisy, degree)
    y_pred = np.polyval(model, x)
    error = MSE_Error(y_true, y_pred)
    errors.append(error)

# Find the best degree
best_degree = degrees[np.argmin(errors)]

# Fit the best polynomial
best_model = fit_polynomial(x, y_noisy, best_degree)
y_best_fit = np.polyval(best_model, x)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y_noisy, color='lightgray', label='Noisy samples', s=20)
plt.plot(x, y_true, color='blue', label='answer')
plt.plot(x, y_best_fit, color='red', label=f'predict (degree {best_degree})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Objective Function and Best Fit Polynomial')
plt.legend()
plt.grid(True)
plt.show()
