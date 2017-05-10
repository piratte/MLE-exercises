from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans as km
from sklearn import metrics


class KMeans:
    def __init__(self, num_of_clusters, max_iter=20):
        self.k = num_of_clusters
        self.centroids = np.array([])
        self.max_iter = max_iter

    def has_changed(self, new_clusters, clusters):
        return (new_clusters != clusters).any()

    def compute_centroids(self, data, clusters):
        new_centroids = data[:self.k]
        for i in range(0, self.k):
            try:
                new_centroids[i] = np.mean(data[clusters == i], axis=0)
            except IndexError:
                print("Error!")
                new_centroids[i] = np.random.randint(0, high=1, size=(1, data.shape[1]))
        return new_centroids

    def update_clustering(self, train_X, centroids):
        # vybrat minimalni vzdalenost od bodu k centroidu ~ predict
        return np.array([self.predict_one(x, centroids) for x in train_X])

    def fit(self, input_data):
        train_X = np.array(input_data)
        # random initialization
        new_clustering = np.random.randint(self.k, size=len(train_X))
        clustering = new_clustering + 1
        centroids = np.random.randint(0, 1, size=(self.k, train_X.shape[1]))
        iteration_number = 0

        while self.has_changed(new_clustering, clustering) and iteration_number < self.max_iter:
            print("Iteration: ", iteration_number)
            clustering = new_clustering
            centroids = self.compute_centroids(train_X, clustering)
            new_clustering = self.update_clustering(train_X, centroids)
            iteration_number += 1

        self.centroids = centroids

    def predict_one(self, one_X, centroids):
        dists = [np.linalg.norm(one_X - cent) for cent in centroids]
        return np.argmin(dists)

    def predict(self, test_X):
        return [self.predict_one(x, self.centroids) for x in test_X]

if __name__ == '__main__':
    data_test = pd.read_csv("data/pamap_easy.test.txt", sep='\t', header=None).values
    data_test_labels = data_test[:, -1]
    data_test = data_test[:, :-1]
    data_train = pd.read_csv("data/pamap_easy.train.txt", sep='\t', header=None).values
    data_train_labels = data_train[:, -1]
    data_train = data_train[:, :-1]

    num_clusters = len(np.unique(np.concatenate((data_test_labels, data_train_labels))))

    algs = [(KMeans(num_clusters, 50), "My K-Means"),
            (km(init='k-means++', n_clusters=num_clusters, n_init=10), "Scikit, k-means++ initialization"),
            (km(init='random', n_clusters=num_clusters, n_init=10), "Scikit, random initialization")]

    for alg in algs:
        estimator = alg[0]
        estimator.fit(np.concatenate((data_train, data_test)))
        estimated_labels = estimator.predict(data_test)
        print("Statistics for ", alg[1])
        print(metrics.homogeneity_score(data_test_labels, estimated_labels),
              metrics.v_measure_score(data_test_labels, estimated_labels),
              metrics.completeness_score(data_test_labels, estimated_labels))
