# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:57:01 2020

@author: binte naeem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("50_Startups.csv")
Country= dataset.loc[dataset.State=="California" , :]
Country_y = Country.iloc[:, -1].values
invest= np.arange(17)
Country_x = invest.reshape(-1, 1)

"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(Country_x, Country_y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
Country_x_poly = poly_reg.fit_transform(Country_x)
poly_reg.fit(Country_x_poly, Country_y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(Country_x_poly, Country_y)

plt.scatter(Country_x, Country_y, color = 'red')
plt.plot(Country_x, lin_reg.predict(Country_x), color = 'green')
plt.title('graph for California (Linear Regression)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()

plt.scatter(Country_x, Country_y, color = 'red')
plt.plot(Country_x, lin_reg_2.predict(poly_reg.fit_transform(Country_x)), color = 'blue')
plt.title('graph for California (Polynomial Regression)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()

Country_x_grid = np.arange(min(Country_x), max(Country_x), 0.1)
Country_x_grid = Country_x_grid.reshape((len(Country_x_grid), 1))
plt.scatter(Country_x, Country_y, color = 'red')
plt.plot(Country_x_grid, lin_reg_2.predict(poly_reg.fit_transform(Country_x_grid)), color = 'blue')
plt.title('Chart for California (Polynomial Regression)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()

lin_reg.predict([[0]])

lin_reg_2.predict(poly_reg.fit_transform([[0]]))


dataset= pd.read_csv("50_Startups.csv")
Country= dataset.loc[dataset.State=="New York" , :]
Country_y = Country.iloc[:, -1].values
invest= np.arange(17)
Country_x = invest.reshape(-1, 1)

"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(Country_x, Country_y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
Country_x_poly = poly_reg.fit_transform(Country_x)
poly_reg.fit(Country_x_poly, Country_y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(Country_x_poly, Country_y)

plt.scatter(Country_x, Country_y, color = 'red')
plt.plot(Country_x, lin_reg.predict(Country_x), color = 'green')
plt.title('graph for New York (Linear Regression)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()

plt.scatter(Country_x, Country_y, color = 'red')
plt.plot(Country_x, lin_reg_2.predict(poly_reg.fit_transform(Country_x)), color = 'blue')
plt.title('graph for New York (Polynomial Regression)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()

Country_x_grid = np.arange(min(Country_x), max(Country_x), 0.1)
Country_x_grid = Country_x_grid.reshape((len(Country_x_grid), 1))
plt.scatter(Country_x, Country_y, color = 'red')
plt.plot(Country_x_grid, lin_reg_2.predict(poly_reg.fit_transform(Country_x_grid)), color = 'blue')
plt.title('Graph for New York(Polynomial Regression)')
plt.xlabel('Investment')
plt.ylabel('Profit')
plt.show()

lin_reg.predict([[0]])

lin_reg_2.predict(poly_reg.fit_transform([[0]]))



