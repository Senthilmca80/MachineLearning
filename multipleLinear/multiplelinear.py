# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:43:04 2020

@author: sk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset=pd.read_csv('50_Startups.csv')
x= dataset.iloc[: , :-1].values
y= dataset.iloc[: , 4].values

#encoding categorical value

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x= LabelEncoder()
x[: ,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding last dummy value trap
x= x[: , 1:]



from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x , y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
