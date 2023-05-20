# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:31:43 2023

@author: Krzysiu
"""
import time
import Data
from RegressionTree import RegressionTree
from LinearRegression import LineralRegression
from PolynominalRegression import PolynominalRegression
from copy import deepcopy

datasets = [
    ['CarPrice_Assignment.csv', 'price']
]
y_label = datasets[0][1]  # tmp
data = Data.get_data(datasets[0][0])

train_data, test_data = Data.split_data(data)

y_train = train_data[y_label].to_numpy()
x_train = train_data['horsepower'].to_numpy()
x_train_2 = train_data['curbweight'].to_numpy()
#x_train = train_data.drop(y_label, axis=1).to_numpy()

y_test = test_data[y_label].to_numpy()
x_test = test_data['horsepower'].to_numpy()
x_test_2 = test_data['curbweight'].to_numpy()
# x_test = test_data.drop(y_label, axis=1).to_numpy()


## test implementation

regressors = [LineralRegression(), PolynominalRegression(), RegressionTree(min_sample_split=3, max_depth=3)]

for regressor in regressors:
    time_0 = time.time()

    regressor.fit(x_train, x_train_2, y_train)

    regressor.predict(x_train, x_train_2)
    regressor.plot(x_test, x_test_2, y_test)
    regressor.print()

    if regressors.index(regressor) == 0 or regressors.index(regressor) == 1:
        x, x_2, y = regressor.GradientDescent(x_train, x_train_2, y_train, x_test, x_test_2, y_test)
        regressor.plot(x, x_2, y)
        regressor.print()

    print(f"time: {time.time() - time_0}\n")
    if isinstance(regressor, RegressionTree):
        regressor.print_tree()
    else:
        regressor.print()
    # regressor.plot(x_test, y_test)