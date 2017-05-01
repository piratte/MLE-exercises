from __future__ import print_function

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from scipy.spatial.distance import sqeuclidean

import numpy as np


if __name__ == '__main__':
    artif_train_X = pd.read_csv("data/artificial_2x_train.tsv", sep='\t', header=None)
    artif_train_y = artif_train_X.iloc[:, -1]
    artif_train_X = artif_train_X.iloc[:, :-1]

    artif_test_X = pd.read_csv("data/artificial_2x_test.tsv", sep='\t', header=None)
    artif_test_y = artif_test_X.iloc[:, -1]
    artif_test_X = artif_test_X.iloc[:, :-1]

    flats_train = pd.read_csv("data/pragueestateprices_train.tsv", sep='\t', header=None)
    flats_train_count = flats_train.shape[0]
    flats_train_y = flats_train.iloc[:, -2]
    flats_train = flats_train.iloc[:, :-2]
    flats_test = pd.read_csv("data/pragueestateprices_test.tsv", sep='\t', header=None)
    flats_test_y = flats_test.iloc[:, -2]
    flats_test = flats_test.iloc[:, :-2]
    flats_all = pd.get_dummies(pd.concat([flats_train, flats_test]), columns=[1, 2, 3, 5, 6, 7])
    flats_train_X, flats_test_X = flats_all.iloc[:flats_train_count, :].values.tolist(), flats_all.iloc[
                                                                                         flats_train_count:,
                                                                                         :].values.tolist()

    classifiers = [(knn(), "knn"), (DecisionTreeRegressor(), "dt"), (AdaBoostRegressor(), "ada")]

    for clsMet, clsName in classifiers:
        preds = clsMet.fit(artif_train_X, artif_train_y).predict(artif_test_X)
        score = list(map(lambda x: sqeuclidean(x[0], x[1]), zip(preds, artif_test_y)))
        print("Score for %s on artif is %f" % (clsName, np.mean(np.array(score))))

        preds = clsMet.fit(flats_train_X, flats_train_y).predict(flats_test_X)
        score = list(map(lambda x: sqeuclidean(x[0], x[1]), zip(preds, flats_test_y)))
        print("Score for %s on flats is %f" % (clsName, np.mean(np.array(score))))