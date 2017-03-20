#!/usr/bin/env python
import numpy as np

# load the MRI scans and metadata
people = np.load('scans.npy')

# open the file containing the description of the data
dataFile = open('data.txt', 'w')

# write a header to the file
dataFile.write('personID,gender,dementia,age\n')

for p in people:
	# for each person write the info about him to the data file
	dataFile.write('{},{},{},{}\n'.format(p['ID'], p['gender'], p['dementia'], p['age']))

# close the file
dataFile.close()
