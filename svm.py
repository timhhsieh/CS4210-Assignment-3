#-------------------------------------------------------------------------
# AUTHOR: Tim Hsieh
# FILENAME: svm.py
# SPECIFICATION: simulating grid search to find combination of four SVM hyperparameters to best predict performance
# FOR: CS 4210- Assignment #3
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c_values = [1, 5, 10, 100]
degree_values = [1, 2, 3]
kernel_values = ["linear", "poly", "rbf"]
decision_function_shape_values = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:, :64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:, -1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

# Initialize variables to keep track of the highest accuracy and corresponding hyperparameters
highest_accuracy = 0
best_hyperparameters = {}

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for c_value in c_values:
    for degree_value in degree_values:
        for kernel_value in kernel_values:
            for decision_function_shape_value in decision_function_shape_values:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, decision_function_shape=decision_function_shape_value)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                # Initialize variables to compute accuracy
                correct_predictions = 0
                total_samples = len(X_test)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                for x_test_sample, y_test_sample in zip(X_test, y_test):
                    prediction = clf.predict([x_test_sample])
                    if prediction[0] == y_test_sample:
                        correct_predictions += 1

                accuracy = correct_predictions / total_samples

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_hyperparameters = {
                        'C': c_value,
                        'degree': degree_value,
                        'kernel': kernel_value,
                        'decision_function_shape': decision_function_shape_value
                    }
                    print(f"Highest SVM accuracy so far: {highest_accuracy:.2f}, Parameters: {best_hyperparameters}")
