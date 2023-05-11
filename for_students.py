import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits import mplot3d

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
y_train = train_data['price'].to_numpy()
x_train = train_data['enginesize'].to_numpy()
x_train_2 = train_data['wheelbase'].to_numpy()

y_test = test_data['price'].to_numpy()
x_test = test_data['enginesize'].to_numpy()
x_test_2 = test_data['wheelbase'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0, 0]

Y = y_train [np.newaxis]
Y = Y.T
Y = np.matrix(Y)

X = np.column_stack((np.ones(len(x_train)), x_train, x_train_2))
X = np.matrix(X)

time_0 = time.time()
theta_best = ((X.T*X).I)*X.T*Y
print('\nClosed-form solution')
print(f'Best theta: {float(theta_best[2])}, {float(theta_best[1])}, {float(theta_best[0])}')
print(f"Time: {time.time() - time_0}")
# TODO: calculate error

MSE = 0
for i in range(0, len(X)):
    MSE += (y_train[i] - float(theta_best[0]) - float(theta_best[1]) * x_train[i] - float(theta_best[2])* x_train_2[i])**2
MSE /= len(X)
print(f'Error: {MSE}\n')

# plot the regression line
fig = plt.figure()
x = np.linspace(min(x_test), max(x_test), 100)
y = np.linspace(min(x_test_2), max(x_test_2), 100)
z = float(theta_best[0]) + float(theta_best[1]) * x + float(theta_best[2]) * y
ax = plt.axes(projection='3d')
ax.set_xlabel('enginesize')
ax.set_ylabel('wheelbase')
ax.set_zlabel('price')
ax.plot3D(x, y, z, 'blue')
ax.scatter(x_test, x_test_2, y_test)
plt.show()

# TODO: standardization
x_std = []
x_std_2 = []
y_std = []
for i in range(0, len(X)):
    z = (x_train[i]-np.mean(x_train))/np.std(x_train)
    x_std.append(z)
    z_2 = (x_train_2[i] - np.mean(x_train_2)) / np.std(x_train_2)
    x_std_2.append(z_2)
    z = (y_train[i]-np.mean(y_train))/np.std(y_train)
    y_std.append(z)

x_mean = np.mean(x_train)
x_mean_2 = np.mean(x_train_2)
y_mean = np.mean(y_train)

x_sttd = np.std(x_train)
x_sttd_2 = np.std(x_train_2)
y_sttd = np.std(y_train)

X_std = np.column_stack((x_std_2, x_std, np.ones(len(x_train))))
X_std = np.matrix(X_std)

y_std = np.array(y_std)
Y_std = y_std [np.newaxis]
Y_std = Y_std.T
Y_std = np.matrix(Y_std)

x_std_test = []
x_std_2_test = []
y_std_test = []

for i in range(0, len(x_test)):
    z = (x_test[i]-x_mean)/x_sttd
    x_std_test.append(z)
    z_2 = (x_test_2[i] - x_mean_2) / x_sttd_2
    x_std_2_test.append(z_2)
    z = (y_test[i]-y_mean)/y_sttd
    y_std_test.append(z)


# TODO: calculate theta using Batch Gradient Descent
η = 0.001
theta_beg = [0, 0, 0]
theta_beg = np.matrix(theta_beg)
theta_beg = theta_beg.T

time_1 = time.time()
m = len(X_std)
for i in range(1, 2500):
    new_theta = theta_beg - η*((2/m)*X_std.T*(X_std*theta_beg-Y_std))
    theta_beg = new_theta

print('Batch Gradient Descent')
print(f'Best theta: {float(theta_beg[2])}, {float(theta_beg[1]), float(theta_beg[0])}')
print(f"Time: {time.time() - time_1}")
# TODO: calculate error

MSE = 0
for i in range(0, len(x_test)):
    MSE += (y_std_test[i] - float(theta_beg[1]) - float(theta_beg[0]) * x_std_test[i])**2
MSE /= len(x_test)
print(f'Error: {MSE}')

# plot the regression line
x = np.linspace(min(x_std_test), max(x_std_test), 100)
y = np.linspace(min(x_std_2_test), max(x_std_2_test), 100)
z = float(theta_beg[2]) + float(theta_beg[1]) * x + float(theta_beg[0]) * y
ax = plt.axes(projection='3d')
ax.set_xlabel('enginesize')
ax.set_ylabel('wheelbase')
ax.set_zlabel('price')
ax.plot3D(x, y, z, 'blue')
ax.scatter(x_std_test, x_std_2_test, y_std_test)
plt.show()
