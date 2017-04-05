#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load the MRI scans and metadata
people = np.load('scans.npy')

# load the IDs of people from the training data
tf = open('train.txt', 'r')
tf.readline()
trainIDs = [l.split(',')[0] for l in tf.readlines()]
tf.close()

# calculate the average measured value in the center third (in each dimension, approximately)
avgs = []
ages = []

for p in people:
	if not p['ID'] in trainIDs:
		continue

	avgs.append(np.mean(p['scan'][40:80, 50:100, 45:95, :]))
	ages.append(p['age'])

# plot the results
plt.scatter(avgs, ages, marker='.', label='Measured values')

# do a least squares fit of a cubic polynomial through the data
p2 = np.poly1d(np.polyfit(avgs, ages, 2))

# plot the resulting curve in the same graph
avgsSpan = max(avgs) - min(avgs)
agesSpan = max(ages) - min(ages)

xp = np.linspace(min(avgs)-0.1 * avgsSpan, max(avgs) + 0.1*avgsSpan, 1000)
plt.plot(xp, p2(xp), '-', c='r', label='Quadratic polynomial fit')

# set the limits of the figure
plt.xlim(min(avgs)-0.05 * avgsSpan, max(avgs) + 0.05*avgsSpan)
plt.ylim(min(ages)-0.05 * agesSpan, max(ages) + 0.05*agesSpan)

# set the labels
plt.xlabel('Average MRI value in the center third of the brain')
plt.ylabel('Age')

# show the legend
plt.legend()

# plt.show()

# save the result to a file
plt.savefig('visualization.png')


