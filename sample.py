#!/usr/bin/python

# Adapted from http://hackmap.blogspot.com/2007/09/k-means-clustering-in-scipy.html

import numpy
import matplotlib
import random
import numpy as np
matplotlib.use('Agg')
from scipy.cluster.vq import *
import pylab
pylab.close()

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

path_to_image = '/Users/margaretwm3/Desktop/kmeans.png'
# generate 3 sets of normally distributed points around
# different means with different variances
# pt1 = numpy.random.normal(1, 0.2, (100,2))
# pt2 = numpy.random.normal(2, 0.5, (300,2))
# pt3 = numpy.random.normal(3, 0.2, (200,2))
# pt1 = numpy.random.normal(1, 0.2, (20,2))
# pt2 = numpy.random.normal(2, 0.5, (10,2))
# pt3 = numpy.random.normal(3, 0.2, (10,2))

# slightly move sets 2 and 3 (for a prettier output)
#pt2[:,0] += 1
#pt3[:,0] -= 0.5

#xy = numpy.concatenate((pt1, pt2, pt3))
	
#res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1])),5)
#dataSet = init_board(300) # create the initial configuration of the board
dataSet = init_board_gauss(300,3)
print dataSet
# kmeans for 5 clusters
res, idx = kmeans2(dataSet,3)

colors = ([([1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1])[i] for i in idx])

# plot colored points
pylab.scatter(dataSet[:,0],dataSet[:,1], c=colors)

# mark centroids as (X)
pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
#print "here"
pylab.savefig(path_to_image)

