import numpy as np
import sys
sys.path.append("..")
from train_test_split import train_test_split as split
from OneNN import OneNearestNeighbour
from sklearn.datasets import load_iris


iris = load_iris()
iris_features = iris['data']
iris_labels = iris['target']
print(iris_features)
print(iris_labels)
iris_X_train, iris_X_test, iris_y_train, iris_y_test = split(iris_features,iris_labels,0.25,True)
print(iris_X_train.shape)
print(iris_X_test.shape)

NN = OneNearestNeighbour()
NN.fit(iris_X_train,iris_y_train)

iris_predictions = NN.predict(iris_X_test)

print(iris_predictions == iris_y_test)
print("Accuracy :", np.mean(iris_predictions == iris_y_test))
