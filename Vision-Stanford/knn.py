import numpy as np
class NN(object):
    def __init__ (self):
        pass

    def train(self, X, y):
        # save the train data as model attributes, it will be required for prediction
        self.X = X
        self.y = y

    def predict(self, examples):
        y_pred = []
        for ex in examples:
            pred = self.__predict__(ex)
            y_pred.append(pred)
        return np.array(y_pred)

    def __predict__(self, ex):
        # compute distances of all trian examples from the test example
        distances = np.sum(np.abs(self.X - ex[:,]), axis=1)
        # find the index of nearest example
        nearest   = np.argmin(distances)
        # return the label of nearest example
        return self.y[nearest]

class KNN(object):
    def __init__ (self):
        pass

    def train(self, X, y):
        # save the train data as model attributes, it will be required for prediction
        self.X = X
        self.y = y

    def predict(self, examples, k):
        y_pred = []
        for ex in examples:
            pred = self.__predict__(ex, k)
            y_pred.append(pred)
        return np.array(y_pred)

    def __predict__(self, ex, k):
        # compute distances of all trian examples from the test example
        distances = np.sum(np.abs(self.X - ex[:,]), axis=1)

        # get the labels K nearest examples
        k_labels  = self.y[np.argsort(distances)[:k]]

        # find the counts of labels in list of k nearest examples
        labels, counts = np.unique(k_labels, return_counts=True)

        # return the label with heighest count
        return labels[np.argmax(counts)]
