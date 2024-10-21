# Linear Regression Analysis on Student Performance with Python

This project applies a Linear Regression model to predict student performance based on several factors. The dataset used comes from student performance records and includes features related to academic history and personal study habits.

The goal of the model is to predict the final grade of a student (`G3`) based on other features.

## Dataset

The dataset used is `student-mat.csv`, which includes various attributes of students. In this project, I focus on the following features:

- `G1`: First period grade
- `G2`: Second period grade
- `G3`: Final grade (target variable)
- `studytime`: Weekly study time (categorical)
- `failures`: Number of past class failures
- `absences`: Number of school absences

## Methodology

Linear Regression is used to predict the final grade (`G3`) of students using the other selected features (`G1`, `G2`, `studytime`, `failures`, `absences`). The steps I used are:

1. **Data Preprocessing**: Read the dataset, select relevant columns, and split it into training and testing sets.
   
2. **Model Training**: I use scikit-learn's `LinearRegression()` to train the model on the training data.

3. **Model Evaluation**: The model's accuracy is evaluated using the testing data, and I print the model's coefficients and intercept to understand the influence of each feature.

4. **Predictions**: The model is used to predict the final grade for the test set, and these predictions are compared to actual values for evaluation.

