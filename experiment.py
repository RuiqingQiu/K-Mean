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
from s_kmean import sequential_kmean

#parameters

# dimension
p = 2

# sqrt of variance
sigma = 0.99

# k is number of clusters
k = 3

# n is number of points to be generated
n = 1000

# init a covariance matrix as identity matrix
identity_matrix = np.identity(p)

# covariance matrix
covariance = sigma * sigma * identity_matrix
d = Data(1000, k, p)
# choose means from

# means are from a normal distribution with center is all 0s and varience N(0,sigma squre identity matrix)
mean = []
for i in range(p):
    mean.append(0)
# choose means, k is the number of centers/cluster
means = d.generate_mult_normal_data(mean, covariance, k)

d.set_true_center(means)

# choose pis, ASA probabolity
PIs = []
for i in range(k):
	# every cluster has the same probability
    PIs.append(1.0/k)

# generate n points from the mixture model
data_set = d.generate_mult_normal_based_prob(PIs, means, covariance, n)

# expriment starts
res1, idx1 = kmeans2(data_set, k)
score = d.error_calculate(res1) / (sigma * sigma * p)
print "kmean score is ", score

# first graph - kmean
colors = ([([1,0,0],[0,1,0],[0,0,1])[i] for i in idx1])
pylab.scatter(data_set[:,0],data_set[:,1], c=colors)
pylab.scatter(d.true_center[:,0],d.true_center[:,1], marker='H', s = 500, linewidths=2)
# o means kmean center
pylab.scatter(res1[:,0],res1[:,1], marker='o', s = 500, linewidths=2, c='none')
# path_to_image = '/Users/margaretwm3/Desktop/kmeans.png'
path_to_image = '/Users/ruiqingqiu/Desktop/kmeans.png'
pylab.savefig(path_to_image)
pylab.clf()

# run sequential kmean
res2, idx2 = sequential_kmean(data_set, k)
score = d.error_calculate(res2) / (sigma * sigma * p)
print "sequential kmean score is ", score

# second graph - sequential kmean
colors = ([([1,0,0],[0,1,0],[0,0,1])[i] for i in idx2])
pylab.scatter(data_set[:,0],data_set[:,1], c=colors)
pylab.scatter(d.true_center[:,0],d.true_center[:,1], marker='H', s = 500, linewidths=2)
# x means sequential centers
pylab.scatter(res2[:,0],res2[:,1], marker='x', s = 500, linewidths=2)
# path_to_image = '/Users/margaretwm3/Desktop/sequential_kmeans.png'
path_to_image = '/Users/ruiqingqiu/Desktop/sequential_kmeans.png'
pylab.savefig(path_to_image)




