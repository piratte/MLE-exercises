#!/usr/bin/env python
import random

# set the seed so we always get the same split
random.seed(42)

# ratio of training data from all the data
trainRatio = 0.8

# open the file containing the description of the data
dataFile = open('hists.txt', 'r')

# load the header line
header = dataFile.readline()

# open the header file
headerFile = open('header.txt', 'w')

# write the header line
headerFile.write(header)

# load the rest of the lines
lines = dataFile.readlines()

# shuffle the lines
random.shuffle(lines)

# get how many lines should be in the training file
trainLines = int(len(lines) * trainRatio)

# write the first X of the shuffled lines to the training file
trainFile = open('train.txt', 'w')
trainFile.writelines(lines[:trainLines])

# write the rest of the lines to the test file
testFile = open('test.txt', 'w')
testFile.writelines(lines[trainLines:])

# close all the opened files
dataFile.close()
headerFile.close()
trainFile.close()
testFile.close()
