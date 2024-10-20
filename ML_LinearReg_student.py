import tensorflow
import keras

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student_performanceData/student/student-mat.csv", sep=";") #sep is a separator

# pd.read_csv() is used to load data from a CSV file into a pandas DataFrame.


# print(data.head()) #Shows the first 5 data in dataframe

# Have seen the data has 33 attributes shown in the website where the data is downloaded from, however, we are not interested in all 33 attributes.

# where G1 G2 G3 are grades in different times.

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head()) #Shows the first 5 data in the new dataframe after picking certain attributes.

predict = "G3"

#Now will set up 2 arrays: 1. Define Attributes, 2. Labels

# 1) X - dropping G3 column, while including the other columns - these are the attributes that might influence the outcome G3.

# 2) y - contains target data G3 - this is what the model will learn to estimate based on the features.
# This allows the model to compare its predictions with the true values and adjust its internal parameters to improve accuracy.

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict]) #Data Array.

#X - Attributes/Feature Array (input data)
#y - Label Array (output data - target we try predict)


#Now we will split these up into 4 variables. 10% used for testing, while 90% for training.

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# The dataset is split into 2 parts: Training set and Testing set.

# Training set: X_train y_train - portion of the data used to train the model (learns the relationship between input features and output label)

# Testing set: X_test y_test - after the model has been trained, its evaluating its results comparing to the testing set (which it hasn't seen before)


# Now need to create a training model
linear = linear_model.LinearRegression() #This is from the sklearn package

linear.fit(X_train, y_train)
accuracy = linear.score(X_test,y_test)
print(accuracy)

#First run got 0.768570.. The prediction on what the students grade (G3) going to be from the given attributes we have.

#Second run got 0.780037.. 


# We can print out the coefficients and y-intercept for the linear regression line.

#This algorithm models the relationship between the dependent variables (labels) and the independent variable G3 (feature):
print('Coefficient: \n', linear.coef_) #Weightings
print('Intercept: \n', linear.intercept_)

# Results: Coefficient:
# [ 0.12363971  0.99950853 -0.16909257 -0.32582331  0.03401172]
# Coeff for each 5 different variables.
# ie. 1 unit increase in G1 means the final grade G3 increases about 0.15 units etc.
# This shows G2 has the highest positive impact on G3 as its weighting is at 0.97.

# Intercept: 
# -1.3721359536392335

# y-intercept shows that when all weightings are 0, G3 would approximate to -1.41.


# We can now use this informaton to predict for the testing data X.
# The predictions array will contain the predicted values for the final grade (G3) based on the test data.
predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])

# This loop iterates through each prediction and compares it with the actual values from the testing set.

# Results: First 5 results
# 12.35485444152939 [14 12  2  0 10] 11
# 16.283899112651955 [14 16  1  0  3] 16
# 18.879510179850183 [18 18  1  0  6] 18
# 12.285864991245942 [14 12  1  0  3] 12
# 11.66638871292939 [12 12  3  0  2] 11

# For the first student, the model predicted a final grade of 12.35 based on the features, while the actual grade was 11.

# For the second student, the prediction was 16.28, while actual grade was 16.

# This shows that the models prediction is very close to the actual results - meaning it can predict really well. This could be due to the high percentage of training data - 90%.