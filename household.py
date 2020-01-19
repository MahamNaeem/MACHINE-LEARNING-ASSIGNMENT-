# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 22:04:42 2020

@author: binte naeem
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("housing price.csv")
x = dataset.iloc[:, :1]
y = dataset.iloc[: , 1:2]


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

y_predic = lin_reg.predict(x_test)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('trained (Linear Regression)')
plt.xlabel('identity')
plt.ylabel('sale price')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Tested (Linear Regression)')
plt.xlabel('identity')
plt.ylabel('sale price')
plt.show()

lin_reg.predict([[3000]])

