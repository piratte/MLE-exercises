from __future__ import print_function

import pandas as pd


artif_train_X = pd.read_csv("data/artificial_2x_train.tsv", sep='\t', header=None)
artif_train_y = artif_train_X.iloc[:, -1]
artif_train_X = artif_train_X.iloc[:, :-1]

artif_test_X = pd.read_csv("data/artificial_2x_test.tsv", sep='\t', header=None)
artif_test_y = artif_test_X.iloc[:, -1]
artif_test_X = artif_test_X.iloc[:, :-1]

flats_train = pd.read_csv("data/pragueestateprices_train.tsv", sep='\t', header=None)
flats_train_count = flats_train.shape[0]
flats_train_y = flats_train.iloc[:, -1]
flats_train = flats_train.iloc[:, :-1]
flats_test = pd.read_csv("data/pragueestateprices_test.tsv", sep='\t', header=None)
flats_test_y = flats_test.iloc[:, -1]
flats_test = flats_test.iloc[:, :-1]
flats_all = pd.get_dummies(pd.concat([flats_train, flats_test]), columns=[1, 2, 3, 5, 6, 7])
flats_train_X, flats_test_X = flats_all.iloc[:flats_train_count, :].values.tolist(), flats_all.iloc[flats_train_count:, :].values.tolist()

# TODO: code least [] method gradient descent, one datapoint at the time
