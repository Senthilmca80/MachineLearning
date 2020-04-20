# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:46:23 2020

@author: SK
"""

# import library 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset=pd.read_csv('salary_data.csv')
x= dataset.iloc[: , :-1].values
y= dataset.iloc[: , -1].values

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x , y,test_size=0.2, random_state=0)


#training simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predict the result

y_pred = regressor.predict(x_test)


plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience ( Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience ( Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
