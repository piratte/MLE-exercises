import sys
from sklearn import svm
import pandas as pd
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

sys.path.append('../HW2/')
import common


if __name__ == '__main__':
    for dataset_name in common.ALL_DATASETS:
        print("============ %s dataset ============" % dataset_name)
        if 'artificial' in dataset_name:
            X_train, y_train, _ = common.load_artificial_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _ = common.load_artificial_data('data/%s_test.csv' % dataset_name)
            cont_dims = []

        else:
            SIZE_ADULT_TRAIN, SIZE_ADULT_TEST = 10000, 1000
            X_train, y_train, _, cont_dims = common.load_adult_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _, _ = common.load_adult_data('data/%s_test.csv' % dataset_name)
            X_train, y_train = X_train[:SIZE_ADULT_TRAIN], y_train[:SIZE_ADULT_TRAIN]
            X_test, y_test = X_test[:SIZE_ADULT_TEST], y_test[:SIZE_ADULT_TEST]

        cat_dims = [dim for dim in range(0, len(X_train[0])) if dim not in cont_dims]
        all_data = pd.get_dummies(pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)]),  columns=cat_dims, drop_first=True)
        X_train_new, X_test_new = all_data.iloc[:len(X_train), :].values.tolist(), all_data.iloc[len(X_train):, :].values.tolist()

        classifiers = [(nb(), "naive bayes"),
                       (knn(), "5-nearest neighbours"),
                       (DecisionTreeClassifier(), "decision tree"),
                       (AdaBoostClassifier(), "ADABoost"),
                       (RandomForestClassifier(), "random forest"),
                       (svm.SVC(), "svm")]

        for clf, clf_name in classifiers:
            clf.fit(X_train_new, y_train)
            predictions = clf.predict(X_test_new)
            print('prediction success of the %s: %0.2f%%' % (clf_name, float(sum([x[0] == x[1] for x in list(zip(predictions, y_test))])*100/float(len(y_test)))))
