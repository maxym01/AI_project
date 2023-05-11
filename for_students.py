import numpy as np
import matplotlib.pyplot as plt
import time

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['curbweight'].to_numpy()
x_train = train_data['price'].to_numpy()

y_test = test_data['curbweight'].to_numpy()
x_test = test_data['price'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]

Y = y_train [np.newaxis]
Y = Y.T
Y = np.matrix(Y)

X = np.column_stack((x_train, np.ones(len(x_train))))
X = np.matrix(X)

time_0 = time.time()
theta_best = ((X.T*X).I)*X.T*Y
print('\nClosed-form solution')
print(f'Best theta: {float(theta_best[1])}, {float(theta_best[0])}')
print(f"Time: {time.time() - time_0}")
# TODO: calculate error

MSE = 0
for i in range(0, len(X)):
    MSE += (y_train[i] - float(theta_best[1]) - float(theta_best[0]) * x_train[i])**2
MSE /= len(X)
print(f'Error: {MSE}\n')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[1]) + float(theta_best[0]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('price')
plt.ylabel('curbweight')
plt.show()

# TODO: standardization
x_std = []
y_std = []
for i in range(0, len(X)):
    z = (x_train[i]-np.mean(x_train))/np.std(x_train)
    x_std.append(z)
    z = (y_train[i]-np.mean(y_train))/np.std(y_train)
    y_std.append(z)

x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

x_sttd = np.std(x_train)
y_sttd = np.std(y_train)

X_std = np.column_stack((x_std, np.ones(len(x_train))))
X_std = np.matrix(X_std)

y_std = np.array(y_std)
Y_std = y_std [np.newaxis]
Y_std = Y_std.T
Y_std = np.matrix(Y_std)

x_std_test = []
y_std_test = []

for i in range(0, len(x_test)):
    z = (x_test[i]-x_mean)/x_sttd
    x_std_test.append(z)
    z = (y_test[i]-y_mean)/y_sttd
    y_std_test.append(z)


# TODO: calculate theta using Batch Gradient Descent
η = 0.001
theta_beg = [0, 0]
theta_beg = np.matrix(theta_beg)
theta_beg = theta_beg.T

time_1 = time.time()
m = len(X_std)
for i in range(1, 10000):
    new_theta = theta_beg - η*((2/m)*X_std.T*(X_std*theta_beg-Y_std))
    theta_beg = new_theta

print('Batch Gradient Descent')
print(f'Best theta: {float(theta_beg[1])}, {float(theta_beg[0])}')
print(f"Time: {time.time() - time_1}")
# TODO: calculate error

MSE = 0
for i in range(0, len(x_test)):
    MSE += (y_std_test[i] - float(theta_beg[1]) - float(theta_beg[0]) * x_std_test[i])**2
MSE /= len(x_test)
print(f'Error: {MSE}')

# plot the regression line
x = np.linspace(min(x_std_test), max(x_std_test), 100)
y = float(theta_beg[1]) + float(theta_beg[0]) * x
plt.plot(x, y)
plt.scatter(x_std_test, y_std_test)
plt.xlabel('price')
plt.ylabel('curbweight')
plt.show()