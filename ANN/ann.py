#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:19:11 2019

@author: bruno
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Churn_Modelling.csv',delimiter=",")


# Encoding categorical data
geo=pd.get_dummies(dataset['Geography'],drop_first="true")
combine = pd.concat([geo, dataset], axis=1)
gender=pd.get_dummies(combine['Gender'])
combine = pd.concat([gender, combine], axis=1)
combine = combine.drop(columns=['Geography','Gender','RowNumber','CustomerId','Surname'])

X = combine.iloc[:,0:11].values
print(len(X[1]))
y = combine.iloc[:,-1].values
print(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense



#Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   