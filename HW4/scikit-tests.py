import sys
from sklearn import svm
import pandas as pd
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.feature_extraction import DictVectorizer
from pprint import pprint

sys.path.append('../HW2/')
import common


if __name__ == '__main__':
    for dataset_name in common.ALL_DATASETS:
        dv = DictVectorizer(sparse=False)
        if 'artificial' in dataset_name:
            X_train, y_train, _ = common.load_artificial_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _ = common.load_artificial_data('data/%s_test.csv' % dataset_name)
            cont_dims = []

        else:
            X_train, y_train, _, cont_dims = common.load_adult_data('data/%s_train.csv' % dataset_name)
            X_test, y_test, _, _ = common.load_adult_data('data/%s_test.csv' % dataset_name)

        cat_dims = [dim for dim in range(0, len(X_train[0])) if dim not in cont_dims]
        all_data = pd.get_dummies(pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)]),  columns=cat_dims, drop_first=True)
        X_train_new, X_test_new = all_data.iloc[:len(X_train), :].values.tolist(), all_data.iloc[len(X_train):, :].values.tolist()

        clf = svm.SVC()
        #clf = nb()

        clf.fit(X_train_new, y_train)
        predictions = []
        for x in X_test_new:
            predictions.append(clf.predict([x])[0])
        print('prediction success %f' % float(sum([x[0] == x[1] for x in list(zip(predictions, y_test))])*100/float(len(y_test))))
