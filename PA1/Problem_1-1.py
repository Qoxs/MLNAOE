
import numpy as np
import matplotlib.pyplot as plt

def generate_objective_func(n_sample, noise_std):
    x = np.linspace(0, 6*np.pi, n_sample)
    normal_noise = np.random.normal(0, noise_std, n_sample)
    y = np.sin(x) + x/2 + normal_noise
    return x, y

def fit_polynomial(x, y, degree):
    return np.polyfit(x, y, degree)

def MSE_Error(y_true, y_predict):
    return np.mean((y_true - y_predict)**2)

# 홀드 아웃 방법

n_repeat = 100  # 추가: 반복 횟수
n_sample = 500
ratio = 0.8
noise_std = 0.5

n_train = int(n_sample * ratio)
n_test = n_sample - n_train

degrees = range(1, 60)

# 결과를 저장할 리스트 초기화
all_train_errors = []
all_test_errors = []
all_biases = []
all_variances = []



# n_repeat 만큼 실험 반복 - 결과의 안정성 증가
for _ in range(n_repeat):
    x, y = generate_objective_func(n_sample, noise_std)
    combined = list(zip(x,y))
    np.random.shuffle(combined)
    x[:], y[:] = zip(*combined)

    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_error = []
    test_error = []
    biases = []
    variances = []

    for degree in degrees:
        model = fit_polynomial(x_train, y_train, degree)
        y_train_pred = np.polyval(model, x_train)
        y_test_pred = np.polyval(model, x_test)
        
        train_error.append(MSE_Error(y_train, y_train_pred))
        test_error.append(MSE_Error(y_test, y_test_pred))
        
        y_true = np.sin(x_test) + x_test/2
        bias = np.mean(y_true - y_test_pred)
        variance = np.var(y_test_pred)
        biases.append(bias**2)
        variances.append(variance)

    all_train_errors.append(train_error)
    all_test_errors.append(test_error)
    all_biases.append(biases)
    all_variances.append(variances)


# 결과 평균 계산
avg_train_error = np.mean(all_train_errors, axis=0)
avg_test_error = np.mean(all_test_errors, axis=0)
avg_biases = np.mean(all_biases, axis=0)
avg_variances = np.mean(all_variances, axis=0)

min_test_error_degree = degrees[np.argmin(avg_test_error)]
print("test-mse가 최소가 되는 degree:", min_test_error_degree)

plt.figure(1, figsize=(10, 6))
plt.plot(degrees, avg_train_error, label='Train_error')
plt.plot(degrees, avg_test_error, label='Test_error')
plt.xlabel('Complexity')
plt.ylabel('Mean_Squared_Error')
plt.title('Overfitting(n_sample=1000)')
plt.legend()
plt.grid()
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def generate_objective_func(n_sample, noise_std):
    x = np.linspace(0, 6*np.pi, n_sample)
    normal_noise = np.random.normal(0, noise_std, n_sample)
    y = np.sin(x) + x/2 + normal_noise
    return x, y

def fit_polynomial(x, y, degree):
    return np.polyfit(x, y, degree)

def MSE_Error(y_true, y_predict):
    return np.mean((y_true - y_predict)**2)

def cross_validate_polynomial(x, y, degrees, cv=5):
    mse_scores = []
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        scores = cross_val_score(model, x.reshape(-1, 1), y, scoring='neg_mean_squared_error', cv=cv)
        mse_scores.append(-scores.mean())
    return mse_scores

n_repeat = 100
n_sample = 500
ratio = 0.8
noise_std = 0.5

n_train = int(n_sample * ratio)
n_test = n_sample - n_train

degrees = range(1, 60)

all_train_errors = []
all_test_errors = []
all_biases = []
all_variances = []
all_cv_scores = []

for _ in range(n_repeat):
    x, y = generate_objective_func(n_sample, noise_std)
    combined = list(zip(x,y))
    np.random.shuffle(combined)
    x[:], y[:] = zip(*combined)

    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    train_error = []
    test_error = []
    biases = []
    variances = []

    # K-fold 교차 검증 수행
    cv_scores = cross_validate_polynomial(x, y, degrees, cv=5)
    all_cv_scores.append(cv_scores)

    for degree in degrees:
        model = fit_polynomial(x_train, y_train, degree)
        y_train_pred = np.polyval(model, x_train)
        y_test_pred = np.polyval(model, x_test)
        
        train_error.append(MSE_Error(y_train, y_train_pred))
        test_error.append(MSE_Error(y_test, y_test_pred))
        
        y_true = np.sin(x_test) + x_test/2
        bias = np.mean(y_true - y_test_pred)
        variance = np.var(y_test_pred)
        biases.append(bias**2)
        variances.append(variance)

    all_train_errors.append(train_error)
    all_test_errors.append(test_error)
    all_biases.append(biases)
    all_variances.append(variances)

avg_train_error = np.mean(all_train_errors, axis=0)
avg_test_error = np.mean(all_test_errors, axis=0)
avg_biases = np.mean(all_biases, axis=0)
avg_variances = np.mean(all_variances, axis=0)
avg_cv_scores = np.mean(all_cv_scores, axis=0)

min_test_error_degree = degrees[np.argmin(avg_test_error)]
min_cv_error_degree = degrees[np.argmin(avg_cv_scores)]

print("test-mse가 최소가 되는 degree:", min_test_error_degree)
print("K-fold CV mse가 최소가 되는 degree:", min_cv_error_degree)

plt.figure(figsize=(10, 6))
plt.plot(degrees, avg_train_error, label='Train_error')
plt.plot(degrees, avg_test_error, label='Test_error')
plt.plot(degrees, avg_cv_scores, label='CV_error')
plt.xlabel('Complexity')
plt.ylabel('Mean_Squared_Error')
plt.title(f'Model Comparison (n_sample={n_sample})')
plt.legend()
plt.grid()
plt.show()
'''