#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:03:39 2023

@author: krzysiu
"""
import numpy as np
import matplotlib.pyplot as plt



class LineralRegression:
    def __init__(self, degree=2):
        self.__theta = None
        self.__degree = degree  # TODO

    def fit(self, x, x_2, y):
        """Train model"""
        self.__standard(x, x_2, y)

    def __standard(self, x, x_2, y):
        """Train model with standard method"""
        Y = y[np.newaxis]
        Y = Y.T
        Y = np.matrix(Y)

        X = np.column_stack((np.ones(len(x)), x, x_2))
        X = np.matrix(X)

        self.theta = ((X.T * X).I) * X.T * Y

    def __standarize(self, x_train, x_train_2, y_train, x_test, x_test_2, y_test):
        x_std = []
        x_std_2 = []
        y_std = []
        for i in range(0, len(x_train)):
            z = (x_train[i] - np.mean(x_train)) / np.std(x_train)
            x_std.append(z)
            z_2 = (x_train_2[i] - np.mean(x_train_2)) / np.std(x_train_2)
            x_std_2.append(z_2)
            z = (y_train[i] - np.mean(y_train)) / np.std(y_train)
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
        Y_std = y_std[np.newaxis]
        Y_std = Y_std.T
        Y_std = np.matrix(Y_std)

        x_std_test = []
        x_std_2_test = []
        y_std_test = []

        for i in range(0, len(x_test)):
            z = (x_test[i] - x_mean) / x_sttd
            x_std_test.append(z)
            z_2 = (x_test_2[i] - x_mean_2) / x_sttd_2
            x_std_2_test.append(z_2)
            z = (y_test[i] - y_mean) / y_sttd
            y_std_test.append(z)

        return X_std, Y_std, x_std_test, x_std_2_test, y_std_test

    def GradientDescent(self, x, x_2, y, xt, xt_2, yt):
        """Train model with Gradient Descent"""
        X, Y, x, x_2, y = self.__standarize(x, x_2, y, xt, xt_2, yt)
        η = 0.001
        self.theta = [0, 0, 0]
        self.theta = np.matrix(self.theta)
        self.theta = self.theta.T

        m = len(X)
        for i in range(1, 2500):
            new_theta = self.theta - η * ((2 / m) * X.T * (X * self.theta - Y))
            self.theta = new_theta

        temp = float(self.theta[2])
        self.theta[2] = float(self.theta[0])
        self.theta[0] = float(temp)

        return x, x_2, y

    def predict(self, sample, sample_2):
        """Make prediction for provided sample"""
        return float(self.theta[0]) + float(self.theta[1]) * sample + float(self.theta[2]) * sample_2

    def plot(self, x_test, x_test_2, y_test):
        """plot a graph with test data"""
        fig = plt.figure()
        x = np.linspace(min(x_test), max(x_test), 100)
        y = np.linspace(min(x_test_2), max(x_test_2), 100)
        z = self.predict(x, y)
        ax = plt.axes(projection='3d')
        ax.set_xlabel('horsepower')
        ax.set_ylabel('curbweight')
        ax.set_zlabel('price')
        ax.plot3D(x, y, z, 'blue')
        ax.scatter(x_test, x_test_2, y_test)
        plt.show()

    def print(self):
        """prints some info"""
        print(f"f(x) = {self.theta[2]}*x_2 + {self.theta[1]}x_1 + {self.theta[0]}")

    def score(self, sample):
        """Score of trained model"""
        # TODO
        pass