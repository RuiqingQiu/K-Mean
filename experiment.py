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
p = 3

# sqrt of variance
sigma = 0.5

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

mean = []
for i in range(k):
    mean.append(0)
# choose means
means = d.generate_mult_normal_data(mean, covariance, k)

d.set_true_center(means)

# choose pis
PIs = []
for i in range(k):
    PIs.append(1.0/k)

# generate n points from the mixture model
data_set = d.generate_mult_normal_based_prob(PIs, means, covariance, n)

num_of_iteration = 100
for i in range(num_of_iteration):
    res, idx = kmeans2(data_set, k)
score = d.error_calculate(res) / (sigma * sigma * p)
print "kmean score is ", score



res, idx = sequential_kmean(data_set, k)
score = d.error_calculate(res) / (sigma * sigma * p)
print "sequential kmean score is ", score




