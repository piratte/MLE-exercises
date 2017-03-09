import csv
import sys


def get_possible_vals(dataset, continues_dimensions=[]):
    vals = []
    #try:
    for val_dimension in range(0, len(dataset[0])):
        vals.append(set([]))
        if val_dimension in continues_dimensions:
            continue
        for item in dataset:
            try:
                vals[val_dimension].add(item[val_dimension])
            except IndexError as e:
                print(e)
                print(len(vals), len(item))
                print(val_dimension, item)
                sys.exit(1)
    return vals


def load_artificial_data(filename):
    data = []
    classes = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        for datapoint in list(lines):
            data.append(datapoint[:-1])
            classes.append(datapoint[-1])

    return data, classes, get_possible_vals(data)


def load_adult_data(filename):
    cont_dims = [0, 2, 4, 10, 11, 12]
    data = []
    classes = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        for datapoint in list(lines):
            if not datapoint or len(datapoint) < 2: continue
            d = []
            for idx, val in enumerate(datapoint):
                if idx in cont_dims:
                    d.append(int(val))
                elif idx == len(datapoint)-1:
                    classes.append(val)
                else:
                    d.append(val)
            data.append(d)

    return data, classes, get_possible_vals(data, cont_dims), cont_dims
