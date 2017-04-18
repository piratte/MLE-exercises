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
#ALL_DATASETS = ['transport_profitability',  'tumor']

TRAIN_PATH_TEMPLATE = '../2017-npfl104/%s/train.txt.gz'
TEST_PATH_TEMPLATE = '../2017-npfl104/%s/test.txt.gz'


def get_classes(data):
    return data[data.columns[:-1]], data[data.columns[-1]]

if __name__ == '__main__':
    for dataset_name in ALL_DATASETS:
        #print("==========Now learning the %s dataset==========" % dataset_name)
        if dataset_name == 'transport_profitability':
            train = pd.read_csv(TRAIN_PATH_TEMPLATE % dataset_name, header=None, usecols=[2, 4, 6, 7, 8, 9, 10, 12, 13, 14], compression='gzip')
            train_X, train_y = get_classes(train)
            test = pd.read_csv(TEST_PATH_TEMPLATE % dataset_name, header=None, usecols=[2, 4, 6, 7, 8, 9, 10, 12, 13, 14], compression='gzip')
            test_X, test_y = get_classes(test)
        elif dataset_name == 'gdelt':
            train = pd.read_csv(TRAIN_PATH_TEMPLATE % dataset_name, header=None, usecols=range(4, 16),
                                compression='gzip')
            train_X, train_y = get_classes(train)

            test = pd.read_csv(TEST_PATH_TEMPLATE % dataset_name, header=None, usecols=range(4, 16), compression='gzip', skipfooter=1)
            test_X, test_y = get_classes(test)
        else:
            train = pd.read_csv(TRAIN_PATH_TEMPLATE % dataset_name, header=None, compression='gzip')
            train_X, train_y = get_classes(train)

            test = pd.read_csv(TEST_PATH_TEMPLATE % dataset_name, header=None, compression='gzip')
            test_X, test_y = get_classes(test)

        orig_train_X, orig_train_y, orig_test_X, orig_test_y = train_X, train_y, test_X, test_y

        #if dataset_name == 'gdelt':
        #    cat_dims = [2, 3]
        #    orig_test_X, orig_test_y = orig_test_X[:-1], orig_test_y[:-1]
        #    all_data = pd.get_dummies(pd.concat([pd.DataFrame(train), pd.DataFrame(test[:-1])]),  columns=cat_dims)
        #    train_X, test_X = all_data.head(train_X.shape[0]), all_data.tail(all_data.shape[0] - train_X.shape[0])
        if dataset_name == 'transport_profitability':
            cat_dims = [6, 7, 12, 13]
            cont_dims = [0, 1, 4, 5, 6]
            all_data = pd.get_dummies(pd.concat([pd.DataFrame(train), pd.DataFrame(test)]),  columns=cat_dims)
            train_X, test_X = all_data.head(train_X.shape[0]), all_data.tail(all_data.shape[0] - train_X.shape[0])
        else:
            cont_dims = list(range(0, orig_train_X.shape[0]))
            cat_dims = []

        classifiers = [(DecisionTreeClassifier(), "scikitDecisionTree"),
                       (nb(), "scikitNaiveBayes"),
                       (NaiveBayesClassifier(continuous_dimensions=[dim_ind for dim_ind in range(0, orig_train_X.shape[0])
                                                                    if dim_ind not in cat_dims]), "myNaiveBayes"),
                       (knn(n_jobs=-1), "scikit5-NN"),
                       (AdaBoostClassifier(), "scikitADABoost"),
                       (RandomForestClassifier(n_estimators=100, n_jobs=-1), "scikitRandomForest"),
                       #(svm.SVC(), "rbf svm"),
                       (svm.SVC(kernel='poly', degree=2), "scikitPolynomialSvm"),
                       (svm.SVC(kernel='linear'), "scikitLinearSvm")]

        classifiers_test = [(DecisionTreeClassifier(), "decision tree"), (NaiveBayesClassifier(continuous_dimensions=cont_dims), "myNaiveBayes")]

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
