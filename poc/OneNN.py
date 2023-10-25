import numpy as np

class OneNearestNeighbour:
    def __init__(self):
        self.training_data = None
        self.training_labels = None

    # training of data onto labels
    def fit(self, X, y):
        self.X_training_data = X
        self.y_training_labels = y
    
    # testing function, classifies using helper method which finds euclidean distance
    def predict(self, X):
        predictions = []
        for point in X:
            predictions.append(self._predict_point(point))
        return predictions
    

    def _predict_point(self, point):
        distances = [self._euclidean_distance(point, x) for x in self.X_training_data]
        # print(distances)
        nearest = np.argmin(distances)

        return self.y_training_labels[nearest]

    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

data = [ (1,2), (2,3), (3,4) ] # 2-dimensional coordinates
labels = [ 'a', 'b', 'a' ]     # labels

nn = OneNearestNeighbour()
nn.fit(data, labels)

test_data = [ (2.5,3.5) ] # Should classify as b - close to (2,3)
#print(nn.predict(test_data))  



    
