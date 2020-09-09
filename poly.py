#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:39:16 2020

@author: admin
"""

#import the appropriate packages to be used in poly regression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set for the coke bottles and their prices
x_train = [[2], [3], [4.4], [10], [20]] #size of coke bottles (/10 to get to litres)
y_train = [[7.99], [8.99], [12.99], [14.99], [20.99]] #prices of coke

# Testing set for the coke bottles and their prices
x_test = [[2], [3.5], [5], [15]] #diamters of pizzas
y_test = [[8.50], [10], [10], [17]] #prices of pizzas

# Train the Linear Regression model and plot a prediction for the line of best fit and regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model, 2 being parabola
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(x_train)
X_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph with appropriate labels on the axis and 
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Coca-Cola prices for different bottles')
plt.xlabel('Milliitres/100')
plt.ylabel('Price in Rands')
plt.axis([0, 25, 5, 25])
plt.grid(True)
plt.scatter(x_train, y_train, c='g')
plt.legend(["Line of best fit", "Line of regression", "Training data"])
plt.show()
