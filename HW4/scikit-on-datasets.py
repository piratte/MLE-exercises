from sklearn import svm
import pandas as pd
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

ALL_DATASETS = ['aspect', 'digits', 'gdelt', 'glass', 'transport_profitability', 'tumor']
#ALL_DATASETS = ['aspect', 'digits',  'glass',  'tumor']

def get_classes(data):
    return data[data.columns[:-1]], data[data.columns[-1]]

if __name__ == '__main__':
    for dataset_name in ALL_DATASETS:
        print("==========Now learning the %s dataset==========" % dataset_name)
        if dataset_name == '':
            train = pd.read_csv('2017-npfl104/%s/train.txt.gz' % dataset_name, header=[0], usecols=range(2, 15), compression='gzip')
            train_X, train_y = get_classes(train)

            test = pd.read_csv('2017-npfl104/%s/test.txt.gz' % dataset_name, header=[0], usecols=range(2, 15), compression='gzip')
            test_X, test_y = get_classes(test)
        else:
            train = pd.read_csv('2017-npfl104/%s/train.txt.gz' % dataset_name, header=None, compression='gzip')
            train_X, train_y = get_classes(train)

            test = pd.read_csv('2017-npfl104/%s/test.txt.gz' % dataset_name, header=None, compression='gzip')
            test_X, test_y = get_classes(test)

        if dataset_name == 'gdelt':
            cat_dims = [2, 3]
            all_data = pd.get_dummies(pd.concat([pd.DataFrame(train), pd.DataFrame(test)]),  columns=cat_dims)
            train_X, test_X = all_data.head(train_X.shape[0]), all_data.tail(all_data.shape[0] - train_X.shape[0])
        elif dataset_name == 'transport_profitability':
            cat_dims = [1, 3, 4, 5, 10, 11]
            all_data = pd.get_dummies(pd.concat([pd.DataFrame(train), pd.DataFrame(test)]),  columns=cat_dims)
            train_X, test_X = all_data.head(train_X.shape[0]), all_data.tail(all_data.shape[0] - train_X.shape[0])

        classifiers = [(nb(), "naive bayes"),
                       (knn(), "5-nearest neighbours"),
                       (DecisionTreeClassifier(), "decision tree"),
                       (AdaBoostClassifier(), "ADABoost"),
                       (RandomForestClassifier(), "random forest"),
                       (svm.SVC(), "rbf svm"),
                       (svm.SVC(kernel='poly', degree=2), "polynomial svm"),
                       (svm.SVC(kernel='linear'), "linear svm")]

        classifiers_test = [(nb(), "naive bayes"), (DecisionTreeClassifier(), "decision tree")]

        for clf, clf_name in classifiers:
            clf.fit(train_X, train_y)
            predictions = clf.predict(test_X)
            print('prediction success of the %s: %0.2f%%' %
                  (clf_name, float(sum([x[0] == x[1] for x in list(zip(predictions, test_y))])*100/float(len(test_y)))))
