import sys
from sklearn import svm
import pandas as pd
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

sys.path.append('../HW2/')
from naive_bayes import NaiveBayesClassifier

#ALL_DATASETS = ['aspect', 'digits', 'gdelt', 'glass', 'transport_profitability', 'tumor']
ALL_DATASETS = ['aspect', 'digits', 'glass', 'transport_profitability', 'tumor']
#ALL_DATASETS = ['aspect', 'digits',  'glass',  'tumor']

TRAIN_PATH_TEMPLATE = '../2017-npfl104/%s/train.txt.gz'
TEST_PATH_TEMPLATE = '../2017-npfl104/%s/test.txt.gz'


def get_classes(data):
    return data[data.columns[:-1]], data[data.columns[-1]]

if __name__ == '__main__':
    for dataset_name in ALL_DATASETS:
        #print("==========Now learning the %s dataset==========" % dataset_name)
        if dataset_name == '':
            train = pd.read_csv(TRAIN_PATH_TEMPLATE % dataset_name, header=[0], usecols=range(2, 15), compression='gzip')
            train_X, train_y = get_classes(train)

            test = pd.read_csv(TEST_PATH_TEMPLATE % dataset_name, header=[0], usecols=range(2, 15), compression='gzip')
            test_X, test_y = get_classes(test)
        else:
            train = pd.read_csv(TRAIN_PATH_TEMPLATE % dataset_name, header=None, compression='gzip')
            train_X, train_y = get_classes(train)

            test = pd.read_csv(TEST_PATH_TEMPLATE % dataset_name, header=None, compression='gzip')
            test_X, test_y = get_classes(test)

        orig_train_X, orig_train_y, orig_test_X, orig_test_y = train_X, train_y, test_X, test_y
        if dataset_name == 'gdelt':
            cat_dims = [2, 3]
            all_data = pd.get_dummies(pd.concat([pd.DataFrame(train), pd.DataFrame(test)]),  columns=cat_dims)
            train_X, test_X = all_data.head(train_X.shape[0]), all_data.tail(all_data.shape[0] - train_X.shape[0])
        elif dataset_name == 'transport_profitability':
            cat_dims = [1, 3, 4, 5, 10, 11]

            all_data = pd.get_dummies(pd.concat([pd.DataFrame(train), pd.DataFrame(test)]),  columns=cat_dims)
            train_X, test_X = all_data.head(train_X.shape[0]), all_data.tail(all_data.shape[0] - train_X.shape[0])
        else:
            cat_dims = []

        classifiers = [(nb(), "scikitNaiveBayes"),
                       (NaiveBayesClassifier(continuous_dimensions=[dim_ind for dim_ind in range(0, orig_train_X.shape[0])
                                                                    if dim_ind not in cat_dims]), "myNaiveBayes"),
                       (knn(), "scikit5-NN"),
                       (DecisionTreeClassifier(), "scikitDecisionTree"),
                       (AdaBoostClassifier(), "scikitADABoost"),
                       (RandomForestClassifier(), "scikitRandomForest"),
                       #(svm.SVC(), "rbf svm"),
                       (svm.SVC(kernel='poly', degree=2), "scikitPolynomialSvm"),
                       (svm.SVC(kernel='linear'), "scikitLinearSvm")]

        classifiers_test = [(nb(), "naive bayes"), (DecisionTreeClassifier(), "decision tree")]

        for clf, clf_name in classifiers:
            if clf_name == "myNaiveBayes":
                clf.fit(orig_train_X.values.tolist(), orig_train_y.values.tolist())
                predictions = clf.predict(orig_test_X.values.tolist())
            else:
                clf.fit(train_X, train_y)
                predictions = clf.predict(test_X)
            #print('prediction success of the %s: %0.2f%%' %
            #     (clf_name, float(sum([x[0] == x[1] for x in list(zip(predictions, test_y))])*100/float(len(test_y)))))
            comment = 'ORIGFEATS' if dataset_name not in ['gdelt', 'transport_profitability'] else 'ONEHOT'
            print('%s\t%s\t%0.2f%%\tadammar\t%s' %
                  (dataset_name, clf_name, float(sum([x[0] == x[1] for x in list(zip(predictions, test_y))])*100/float(len(test_y))), comment))
