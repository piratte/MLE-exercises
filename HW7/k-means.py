from __future__ import print_function

import random

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale


class KMeans:
    def __init__(self, num_of_clusters):
        self.k = num_of_clusters
        self.centroids = []

    def has_changed(self, new_clusters, clusters):
        return set(new_clusters) != set(clusters)

    def compute_centroids(self, clusters):
        pass

    def update_clustering(self, train_X, centroids):
        pass

    def fit(self, train_X):
        # random initialization
        new_clusters = random.sample(train_X, self.k)
        clusters = random.sample(train_X, self.k)

        while self.has_changed(new_clusters, clusters):
            clusters = new_clusters
            centroids = self.compute_centroids(clusters)
            new_clusters = self.update_clustering(train_X, centroids)

        self.centroids = centroids

    def predict_one(self, one_X, centroids):
        dists = [np.linalg.norm(one_X - cent) for cent in centroids]
        return np.argmin(dists)

    def predict(self, test_X):
        return [self.predict_one(x, self.centroids) for x in test_X]

if __name__ == '__main__':
    digits = load_digits()
    data = scale(digits.data)

    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target

