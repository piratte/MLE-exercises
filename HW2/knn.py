from __future__ import print_function
import common
import math
from collections import Counter

NUM_NEIGHBOURS = 5


class KnnClassifier:
    def __init__(self, n=1, diff_class_penalty=10):
        self.data = []
        self.cls = []
        self.conts = []
        self.n = n
        self.diff_class_dist = diff_class_penalty

    def train(self, data, classes, continuous_dimensions=[]):
        self.data = data
        self.cls = classes
        self.conts = continuous_dimensions

    def predict(self, data):
        result = []
        for d in data:
            result.append(self.predict_one(d))
        return result

    def predict_one(self, datapoint):
        # go thru all data, find the closest n
        neighbours = []
        for idx, dat in enumerate(self.data):
            if len(neighbours) < self.n:
                neighbours.append((self.get_distance(datapoint, dat), self.cls[idx]))
            else:
                distance = self.get_distance(datapoint, dat)
                max_dist, ind_max = self.get_furthest_neighbour(neighbours)
                if distance < max_dist:
                    neighbours.pop(ind_max)
                    neighbours.append((distance, self.cls[idx]))

        # pick the most common class
        classes = [cls for (dist, cls) in neighbours]
        return Counter(classes).most_common(1)[0][0]

    def get_distance(self, datapoint, dat):
        dist = 0
        for idx, val in enumerate(datapoint):
            if idx in self.conts: # continuous dimension
                dist += pow(val - dat[idx], 2)
            else: # class dimension
                if val != dat[idx]:
                    dist += pow(self.diff_class_dist, 2)
        return math.sqrt(dist)

    @staticmethod
    def get_furthest_neighbour(neighbours):
        m, id_m = neighbours[0][0], 0
        for idx, val in enumerate(neighbours):
            if m < val[0]:
                m, id_m = val[0], idx
        return m, id_m


if __name__ == '__main__':
    for dataset_name in common.ALL_DATASETS:
        if 'artificial' in dataset_name:
            X_train, y_train, _ = common.load_artificial_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _ = common.load_artificial_data('data/%s_test.csv' % dataset_name)
            cont_dims = []
        else:
            SIZE_ADULT_TRAIN, SIZE_ADULT_TEST = 3000, 300
            X_train, y_train, _, cont_dims = common.load_adult_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _, _ = common.load_adult_data('data/%s_test.csv' % dataset_name)
            X_train, y_train = X_train[:SIZE_ADULT_TRAIN], y_train[:SIZE_ADULT_TRAIN]
            X_test, y_test = X_test[:SIZE_ADULT_TEST], y_test[:SIZE_ADULT_TEST]
            print("Using %d training points and %d test points from the adult dataset for performance sake" %
                  (SIZE_ADULT_TRAIN, SIZE_ADULT_TEST))

        cls = KnnClassifier(NUM_NEIGHBOURS)
        cls.train(X_train, y_train, continuous_dimensions=cont_dims)
        res = cls.predict(X_test)
        correct = len(list(filter(lambda x: x[0] == x[1], zip(res, y_test))))
        print("%d%% correct predictions on %s dataset" % (correct*100/len(res), dataset_name))
