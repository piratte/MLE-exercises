import math
import common
from pprint import pprint

class NaiveBayesClassifier:
    SUMMARY_MEAN_IDX = 0
    SUMMARY_STDEV_IDX = 1

    def __init__(self):
        self.continuous_attributes = []
        self.class_summaries = {}

    @staticmethod
    def mean(data):
        return sum(data)/float(len(data))

    def std_dev(self, data):
        avg = self.mean(data)
        variance = sum([pow(x-avg, 2) for x in data])/float(len(data)-1)
        return math.sqrt(variance)

    @staticmethod
    def calculate_probability_for_continuous(x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean, 2)/(2*pow(stdev, 2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    @staticmethod
    def calculate_probability_for_categorical(val, probs):
        if val in probs:
            return probs[val]
        else:
            return 0

    @staticmethod
    def split_by_class(data, classes):
        res = {}
        for idx, datum in enumerate(data):
            cur_class = classes[idx]
            if cur_class not in res:
                res[cur_class] = []
            res[cur_class].append(datum)
        return res

    @staticmethod
    def get_probabilities_per_category(attribute_vals):
        result = {}
        num_of_all = 0
        for val in attribute_vals:
            if val not in result:
                result[val] = 0
            result[val] += 1
            num_of_all += 1
        for val, cnt in result.items():
            result[val] = cnt/float(num_of_all)
        return result

    def summarize(self, instances):
        grouped_data = zip(*instances)
        summaries = []
        for att_idx, attribute_vals in enumerate(grouped_data):
            if att_idx in self.continuous_attributes:
                summary = (self.mean(attribute_vals), self.std_dev(attribute_vals))
            else:
                summary = self.get_probabilities_per_category(attribute_vals)
            summaries.append(summary)
        return tuple(summaries)

    def summarize_by_class(self, dataset, classes):
        separated = self.split_by_class(dataset, classes)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances)
        # pprint(summaries)
        return summaries

    def train(self, train_data, classes, continuous_dimensions):
        self.continuous_attributes = continuous_dimensions
        self.class_summaries = self.summarize_by_class(train_data, classes)

    def predict_one(self, datapoint):
        class_probs = {}
        for class_name, summary_per_attributes in self.class_summaries.items():
            class_probs[class_name] = float(0)
            for att_idx, att_summary in enumerate(summary_per_attributes):
                if att_idx in self.continuous_attributes:
                    prob = self.calculate_probability_for_continuous(
                        datapoint[att_idx],
                        att_summary[self.SUMMARY_MEAN_IDX],
                        att_summary[self.SUMMARY_STDEV_IDX]
                    )
                else:
                    prob = self.calculate_probability_for_categorical(datapoint[att_idx], att_summary)
                class_probs[class_name] += math.log(2 + prob, 2) if prob > 0 else 0
        # pick class with max probability
        max_prob, max_class = -1, ""
        for classValue, probability in class_probs.items():
            if not max_class or probability > max_prob:
                max_prob = probability
                max_class = classValue
        return max_class

    def predict(self, data):
        result = []
        for d in data:
            result.append(self.predict_one(d))
        return result


if __name__ == '__main__':
    for dataset_name in common.ALL_DATASETS:
        if 'artificial' in dataset_name:
            X_train, y_train, _ = common.load_artificial_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _ = common.load_artificial_data('data/%s_test.csv' % dataset_name)
            cont_dims = []
        else:
            X_train, y_train, _, cont_dims = common.load_adult_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _, _ = common.load_adult_data('data/%s_test.csv' % dataset_name)

        cls = NaiveBayesClassifier()
        cls.train(X_train, y_train, continuous_dimensions=cont_dims)
        res = cls.predict(X_test)
        correct = len(list(filter(lambda x: x[0] == x[1], zip(res, y_test))))
        print("%d%% correct predictions on %s dataset" % (correct*100/len(res), dataset_name))
