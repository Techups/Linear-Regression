import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Generate random data-set
np.random.seed(0)
x=np.random.rand(100,1)
y=2+3*x+np.random.rand(100,1)

#Scikit-learn implementation

#Model initialization
regression_model= LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.4)

#fit the data(train the model)
regression_model.fit(x_train,y_train)

#Predict 
y_predicted = regression_model.predict(x_test)
print("misclassified: ",(y_test != y_predicted).sum())

#Model evaluation
rmse = mean_squared_error(y_test,y_predicted)
r2= r2_score(y_test,y_predicted)

#Printing values
print("Slope: ", regression_model.coef_)
print("Intercept: ", regression_model.intercept_)
print("Root Mean Squared: ", rmse)
print("R2 score: ", r2)

#plotting values

#data points
plt.scatter(x_test,y_test,s=10)
plt.xlabel('x')
plt.ylabel('y')

#Predicted values
plt.plot(x_test,y_predicted,color='r')
plt.show()
