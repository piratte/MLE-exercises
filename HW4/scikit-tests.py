import sys
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint

sys.path.append('../HW2/')
import common


if __name__ == '__main__':
    for dataset_name in common.ALL_DATASETS:
        if 'artificial' in dataset_name:
            X_train, y_train, _ = common.load_artificial_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _ = common.load_artificial_data('data/%s_test.csv' % dataset_name)
            cont_dims = []
            enc = OneHotEncoder(categorical_features='all', handle_unknown='ignore')
        else:
            X_train, y_train, _, cont_dims = common.load_adult_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _, _ = common.load_adult_data('data/%s_test.csv' % dataset_name)
            enc = OneHotEncoder(categorical_features=[dim for dim in range(0,len(X_train[0])) if dim not in cont_dims], handle_unknown='ignore')

        pprint(X_train)
        enc.fit(X_train)
        X_train = enc.transform(X_train)
        X_test = enc.transform(X_test)

        #clf = svm.SVC()
        clf = nb()
        clf.fit(X_train, y_train)
        predictions = []
        for x in X_test:
            predictions.append(clf.predict(x))
        print('prediction success %f' % sum(map(lambda x: x[0] == x[1], zip(predictions, y_test)))*100/float(len(y_test)))
