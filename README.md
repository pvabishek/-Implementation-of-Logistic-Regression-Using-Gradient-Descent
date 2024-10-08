# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
```
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.
```
## Program:
 ``````
 /*
 Program to implement the the Logistic Regression Using Gradient Descent.
 Developed by: ABISHEK PV
 RegisterNumber:  212222230003
 */
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 from math import log  # Import log function
 # Use a raw string to handle the file path or replace backslashes with forward slashes
 dataset = pd.read_csv('/content/Placement_Data.csv')
 # Alternatively:
 # dataset = pd.read_csv('C:/Users/admin/Downloads/Placement_Data.csv')
 # Dropping unnecessary columns
 dataset = dataset.drop('sl_no', axis=1)
 dataset = dataset.drop('salary', axis=1)
 # Label encoding categorical features
 from sklearn.preprocessing import LabelEncoder
 le = LabelEncoder()
 # Converting columns to categorical and encoding
 categorical_columns = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specia
 for col in categorical_columns:
 dataset[col] = dataset[col].astype('category')
 dataset[col] = dataset[col].cat.codes
 # Define X and Y
X = dataset.iloc[:, :-1].values
 Y = dataset.iloc[:, -1].values
 # Initialize theta
 theta = np.random.randn(X.shape[1])
 # Sigmoid function
 def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 # Loss function (using log imported from math)
 def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
 # Gradient descent function
 def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
 # Train the model using gradient descent
 theta = gradient_descent(theta, X, Y, alpha=0.01, num_iterations=1000)
 # Prediction function
 def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h > 0.5, 1, 0)
    return y_pred
 # Predict and calculate accuracy
 y_pred = predict(theta, X)
 accuracy = np.mean(y_pred.flatten() == Y)
 print('Accuracy:', accuracy)
 print('Predicted:', y_pred)
 print('Actual:', Y)
 # Predict for new inputs
 xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
 y_prednew = predict(theta, xnew)
 print('New prediction:', y_prednew)
 xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
 y_prednew = predict(theta, xnew)
 print('New prediction:', y_prednew)
```````
## OUTPUT
![image](https://github.com/user-attachments/assets/d92ddadd-453b-43bf-ba10-c4dfd6d8619b)

## RESULT
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
