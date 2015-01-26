#!/usr/bin/python

# Adapted from http://hackmap.blogspot.com/2007/09/k-means-clustering-in-scipy.html

import numpy
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
matplotlib.use('Agg')
from scipy.cluster.vq import *
from scipy.spatial import distance
import pylab
pylab.close()

true_center = []
#path_to_image = '/Users/margaretmw3/Desktop/kmeans.png'
path_to_image = '/Users/ruiqingqiu/Desktop/kmeans.png'

# Taking in k clusters, true_center list, and the result center list
def error_calculate(k,true_center,res):
    error = 0.0
    for i in range(k):
        min_dis = 1000000
        for j in range(3):
            dst = distance.euclidean(true_center[i],res[j])
            if dst < min_dis:
                min_dis = dst
        error += min_dis
    print "the error rate is: ", error
    return error

def plot_graph():
    colors = ([([1,0,0],[0,1,0],[0,0,1])[i] for i in idx])
    # plot colored points
    pylab.scatter(dataSet[:,0],dataSet[:,1], c=colors)

    # mark centroids as (X)
    pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
    pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
    pylab.savefig(path_to_image)



def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        true_center.append([c[0],c[1]]) # remember the true center to calculate error
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    #print true_center
    return X

num_of_iteration = 1000
#res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1])),5)
dataSet = init_board_gauss(300,3)
error_list = []
index = []
for i in range(num_of_iteration):
    res, idx = kmeans2(dataSet,3)
    error_list.append(error_calculate(3,true_center,res))
    index.append(i)


plt.plot(index,error_list,'ro')
#plt.ylabel("error rate")
plt.show()
plot_graph()


