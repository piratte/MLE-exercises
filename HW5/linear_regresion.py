from __future__ import print_function

import pandas as pd
import sys
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from scipy.spatial.distance import sqeuclidean

import numpy as np


class LinearModel:
    def __init__(self, learning_rate, max_epochs):
        self.coef = np.array([])
        self.debug = False
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict_one(self, test_x):
        #if self.debug: print(test_x, self.coef)
        #print(test_x, self.coef)
        #print(np.dot(self.coef, np.insert(test_x, [0], [1])))
        return np.dot(self.coef, np.insert(test_x, [0], [1]))

    def predict(self, test_x):
        self.debug = True
        return np.array([self.predict_one(x) for x in test_x])
        #return list(map(self.predict_one, test_x))

    def fit(self, in_train_x, train_y):
        train_x = np.array(in_train_x)
        dim = len(train_x[0])
        self.coef = np.zeros(dim+1)
        for epoch in range(self.max_epochs):
            sum_error = 0.0
            for datapoint, target in zip(train_x, train_y):
                # print(dim)
                print(self.predict_one(datapoint), target)
                error = self.predict_one(datapoint) - target
                self.coef[0] -= self.learning_rate * error
                for i in range(dim):
                    self.coef[i+1] -= self.learning_rate * error * datapoint[i]
                sum_error += np.float_power(error, 2)
            if sum_error < 0.0001:
                break

if __name__ == '__main__':
    artif_train_X = pd.read_csv("data/artificial_2x_train.tsv", sep='\t', header=None)
    artif_train_y = artif_train_X.iloc[:, -1].values.tolist()
    artif_train_X = artif_train_X.iloc[:, :-1].values.tolist()

    artif_test_X = pd.read_csv("data/artificial_2x_test.tsv", sep='\t', header=None)
    artif_test_y = artif_test_X.iloc[:, -1].values.tolist()
    artif_test_X = artif_test_X.iloc[:, :-1].values.tolist()

    flats_train = pd.read_csv("data/pragueestateprices_train.tsv", sep='\t', header=None)
    flats_train_count = flats_train.shape[0]
    flats_train_y = flats_train.iloc[:, -2]
    flats_train = flats_train.iloc[:, :-2]
    flats_test = pd.read_csv("data/pragueestateprices_test.tsv", sep='\t', header=None)
    flats_test_y = flats_test.iloc[:, -2]
    flats_test = flats_test.iloc[:, :-2]
    flats_all = pd.get_dummies(pd.concat([flats_train, flats_test]), columns=[1, 2, 3, 5, 6, 7])
    flats_train_X, flats_test_X = flats_all.iloc[:flats_train_count, :].values.tolist(), flats_all.iloc[flats_train_count:, :].values.tolist()

    print(len(flats_train_X[0]), len(flats_test_X[0]))

    cls = LinearModel(0.001, 1000)
    cls.fit(artif_train_X, artif_train_y)
    preds = cls.predict(artif_test_X)
    print(preds)
    score = list(map(lambda x: sqeuclidean(x[0], x[1]), zip(preds, artif_test_y)))
    print("Score for %s on artif is %f" % ("mine", np.mean(np.array(score))))
