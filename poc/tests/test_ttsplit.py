import numpy as np
import sys
sys.path.append("..")
from train_test_split import train_test_split as split
from OneNN import OneNearestNeighbour
from sklearn.datasets import load_iris

# loading iris data using sklearn
iris = load_iris()
iris_features = iris['data']
iris_labels = iris['target']

# printing loaded features and labels
print(iris_features)
print(iris_labels)

#splitting data into training and test set using train_test_split
iris_X_train, iris_X_test, iris_y_train, iris_y_test = split(iris_features,iris_labels,0.25,True)

#printing sizes of training and test sets to demonstrate ratio
print(iris_X_train.shape)
print(iris_X_test.shape)

#fitting training data
NN = OneNearestNeighbour()
NN.fit(iris_X_train,iris_y_train)

#predicting labels for test data
iris_predictions = NN.predict(iris_X_test)

#printing correct predictions with accuracy value
print(iris_predictions == iris_y_test)
print("Accuracy :", np.mean(iris_predictions == iris_y_test))
