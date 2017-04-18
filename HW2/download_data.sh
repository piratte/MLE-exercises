#!/usr/bin/env bash

mkdir data
cd data

wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -O adult_test.csv
wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O adult_train.csv

wget http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/artificial_objects.tgz -O dataset.tgz
tar zxf dataset.tgz
rm -f dataset.tgz