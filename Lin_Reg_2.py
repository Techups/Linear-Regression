# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:57:57 2019

@author: user
"""
#This is Linear Regression without splitting the data into train and test
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

#Generate random data-set
np.random.seed(0)
x=np.random.rand(100,1)
y=2+3*x+np.random.rand(100,1)

#Scikit-learn implementation

#Model initialization
regression_model= LinearRegression()

#fit the data(train the model)
regression_model.fit(x,y)

#Predict 
y_predicted = regression_model.predict(x)
print("misclassified: ",(y != y_predicted).sum())

#Model evaluation
rmse = mean_squared_error(y,y_predicted)
r2= r2_score(y,y_predicted)

#Printing values
print("Slope: ", regression_model.coef_)
print("Intercept: ", regression_model.intercept_)
print("Root Mean Squared: ", rmse)
print("R2 score: ", r2)

#plotting values

#data points
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')

#Predicted values
plt.plot(x,y_predicted,color='r')
plt.show()