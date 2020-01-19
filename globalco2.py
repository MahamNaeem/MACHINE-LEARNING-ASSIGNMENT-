# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:53:24 2020

@author: binte naeem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("global_co2.csv")
x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]


"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'green')
plt.title('production of Carbondioxide (Linear Regression)')
plt.xlabel('year')
plt.ylabel('predicted production')
plt.show()

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title( 'production of Carbondioxide (Polynomial Regression)')
plt.xlabel('year')
plt.ylabel('predicted production')
plt.show()

"""x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('production of Carbondioxide (Polynomial Regression)')
plt.xlabel('year')
plt.ylabel('predicted production')
plt.show()"""

lin_reg.predict([[2011]])

lin_reg_2.predict(poly_reg.fit_transform([[2011]]))

lin_reg.predict([[2012]])

lin_reg_2.predict(poly_reg.fit_transform([[2012]]))

lin_reg.predict([[2013]])

lin_reg_2.predict(poly_reg.fit_transform([[2013]]))

