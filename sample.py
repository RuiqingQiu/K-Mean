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
from data_generate import Data

true_center = []
path_to_image = '/Users/ruiqingqiu/Desktop/kmeans.png'
#path_to_image = '/Users/margaretwm3/Desktop/kmeans.png'
'''
# Taking in k clusters, true_center list, and the result center list
def error_calculate(k,true_center,res):
    error = 0.0
    for i in range(k):
        min_dis = 1000000
        for j in range(k):
            dst = distance.euclidean(true_center[i],res[j])
            if dst < min_dis:
                min_dis = dst
        error += min_dis
    print "the error rate is: ", error
    return error
'''
def plot_graph():
    colors = ([([1,0,0],[0,1,0],[0,0,1])[i] for i in idx])
    # plot colored points
    pylab.scatter(dataSet[:,0],dataSet[:,1], c=colors)

    # mark centroids as (X)
    pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
    pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
    pylab.savefig(path_to_image)

"""
def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X
def init_board_gauss(N, k):
    # average number of points needed for each cluster, total points / num of clusters
    n = float(N)/k
    X = []
    # generate k centers
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        true_center.append([c[0],c[1]]) # remember the true center to calculate error

        # s measures how big the cluster is
        s = random.uniform(0.05,0.5)
        # s = 0.05
        x = []
        # generate points for the cluster until it has n points
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        # Put the points into the list
        X.extend(x)
    # convert the list ot np array type and return
    X = np.array(X)[:N]
    #print true_center
    return X
"""
num_of_iteration = 100
#res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1])),5)
cluster_number = 3
data = Data(1000,cluster_number,2)
dataSet = data.init_board_gauss()
error_list = []
index = []
for i in range(num_of_iteration):
    #print dataSet
    res, idx = kmeans2(dataSet,3)
    #print res
    #print idx
    error_list.append(data.error_calculate(res))
    index.append(i)
error_list2 = []
if 5 < cluster_number:
    for i in range(num_of_iteration):
        res, idx = kmeans2(dataSet,5)
        error_list2.append(data.error_calculate(res))
error_list3 = []
if 10 < cluster_number:
    for i in range(num_of_iteration):
        res, idx = kmeans2(dataSet,10)
        error_list3.append(data.error_calculate(res))

"""
#fig = plt.figure()
#p1 = fig.add_subplot(211)
plt.plot(error_list)
plt.plot(error_list2)
plt.plot(error_list3)
#plt.legend(['3','5','10'],loc='upper left')
#p2 = fig.add_subplot(212)
#p2.plot(index,error_list,'ro')
#plt.ylabel("error rate")
plt.show()
plot_graph()
"""
c = []
num_of_center = 3
for j in range(num_of_center):
    c.append([random.uniform(-1,1),random.uniform(-1,1)])
s = 0.1
covariance = [[0.5, 0],[0,0.5]]
#data_set = data.generate_mult_normal_data(c, covariance, 500)

prob = [0.7, 0.2, 0.1]

data_set = data.generate_mult_normal_based_prob(prob, c, covariance, 500)
for i in range(num_of_iteration):
    res, idx = kmeans2(data_set, 3)

colors = ([([1,0,0],[0,1,0],[0,0,1])[i] for i in idx])
pylab.scatter(data_set[:,0],data_set[:,1], c=colors)

pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
pylab.savefig(path_to_image)


pylab.show()

