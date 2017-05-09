from __future__ import print_function

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics


class KMeans:
    def __init__(self, num_of_clusters):
        self.k = num_of_clusters
        self.centroids = np.array([])

    def has_changed(self, new_clusters, clusters):
        return set(new_clusters) != set(clusters)

    def compute_centroids(self, data, clusters):
        new_centroids = np.zeros((self.k, 1))
        for i in range(0, self.k):
            new_centroids[i] = np.mean(data[clusters==i])
        return new_centroids

    def update_clustering(self, train_X, centroids):
        # vybrat minimalni vzdalenost od bodu k centroidu ~ predict
        return [self.predict_one(x, centroids) for x in train_X]

    def fit(self, input_data):
        train_X = np.array(input_data)
        # random initialization
        new_clustering = np.random.randint(self.k, size=len(train_X))
        clustering = np.random.randint(self.k, size=len(train_X))
        centroids = np.random.randint(0,1,size=(self.k, train_X.shape[1]))

        while self.has_changed(new_clustering, clustering):
            clustering = new_clustering
            centroids = self.compute_centroids(train_X, clustering)
            new_clustering = self.update_clustering(train_X, centroids)

        self.centroids = centroids

    def predict_one(self, one_X, centroids):
        dists = [np.linalg.norm(one_X - cent) for cent in centroids]
        return np.argmin(dists)

    def predict(self, test_X):
        return [self.predict_one(x, self.centroids) for x in test_X]

if __name__ == '__main__':
    digits = load_digits()
    digi_data = scale(digits.data)

    n_samples, n_features = digi_data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target

    perm = np.random.permutation(n_samples)
    labels, digi_data = labels[perm], digi_data[perm]

    estimator = KMeans(n_digits)
    estimator.fit(digi_data)
    estimated_labels = estimator.predict(digi_data)
    print(metrics.homogeneity_score(labels, estimated_labels),
          metrics.v_measure_score(labels, estimated_labels),
          metrics.completeness_score(labels, estimated_labels))
