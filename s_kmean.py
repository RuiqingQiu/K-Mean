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
path_to_image = '/Users/ruiqingqiu/Desktop/s_kmeans.png'
#path_to_image = '/Users/margaretwm3/Desktop/s_kmeans.png'

def plot_graph():
    colors = ([([1,0,0],[0,1,0],[0,0,1])[i] for i in idx])
    # plot colored points
    pylab.scatter(dataSet[:,0],dataSet[:,1], c=colors)

    # mark centroids as (X)
    pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
    pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
    pylab.savefig(path_to_image)

def sequential_kmean(data, k):
    kmean_center = []
    idx = []
    for i in range(0, len(data)):
        idx.append(0)
    # set count n1 ... nk to 1
    counts = []
    for i in range(k):
        counts.append(1)

    # set the first seen k points to the kmean center
    for i in range(k):
        kmean_center.append(data[i])
        idx[i] = i   # remember which cluster
    kmean_center_np = np.array(kmean_center)
    index = k
    # look at the next point, and do one pass through the dataset
    while index < len(data):
        # print index
        current_point = data[index]
        min_dist = 100000.0
        replace_position = 0

        # find the distance to the nearest kmean center
        for i in range(k):
            dist = distance.euclidean(kmean_center_np[i], current_point)
            if dist < min_dist:
                min_dist = dist
                replace_position = i
        # index is the current point
        idx[index] = replace_position
        index = index + 1
        # update the kmean center accordingly
        kmean_center_np[replace_position] = ((counts[replace_position] * kmean_center_np[replace_position]) + current_point) / (counts[replace_position] + 1.0)
        # update the countes -> increase the weights
        counts[replace_position] = counts[replace_position] + 1
    return np.array(kmean_center), np.array(idx)

'''
num_of_iteration = 5
cluster_number = 3
d = Data(1000,cluster_number,2)
dataSet = d.init_board_gauss()
error_list_s = []
error_list = []
#index = []
for i in range(num_of_iteration):
    res, idx = sequential_kmean(dataSet,3)
    # print idx
    error_list_s.append(d.error_calculate(res))
    res, idx = kmeans2(dataSet,3)
    error_list.append(d.error_calculate(res))
    #index.append(i)

plt.plot(error_list_s)
plt.plot(error_list)
plt.legend(['s_kmean','kmean'],loc='upper left')
plt.show()
#plot_graph()
'''
