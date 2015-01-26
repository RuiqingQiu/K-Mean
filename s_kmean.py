import numpy
import matplotlib
import random
import numpy as np
matplotlib.use('Agg')
from scipy.cluster.vq import *
from scipy.spatial import distance
import pylab
pylab.close()

true_center = []
#path_to_image = '/Users/ruiqingqiu/Desktop/kmeans.png'
path_to_image = '/Users/margaretwm3/Desktop/kmeans.png'

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

def sequential_kmean(data, k):
    kmean_center = []
    idx = []
    for i in range(0, len(data)):
        idx.append(0)
    # set count n1 ... nk to 1
    counts = []
    for i in range(k):
        counts.append(1)
    for i in range(k):
        kmean_center.append(data[i])
        idx[i] = i
    kmean_center_np = np.array(kmean_center)
    index = k
    while index < len(data):
        print index
        current_point = data[index]
        min_dist = 100000.0
        replace_position = 0
        for i in range(k):
            dist = distance.euclidean(kmean_center_np[i], current_point)
            if dist < min_dist:
                min_dist = dist
                replace_position = i
        idx[index] = replace_position
        index = index + 1
        kmean_center_np[replace_position] = ((counts[replace_position] * kmean_center_np[replace_position]) + current_point) / (counts[replace_position] + 1.0)
        counts[replace_position] = counts[replace_position] + 1
    return np.array(kmean_center), np.array(idx)

num_of_iteration = 1
#res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1])),5)
cluster_number = 3
dataSet = init_board_gauss(1000,cluster_number)
error_list = []
index = []
for i in range(num_of_iteration):
    res, idx = sequential_kmean(dataSet,3)
    print idx
    error_list.append(error_calculate(3,true_center,res))
    res, idx = kmeans2(dataSet,3)
    error_list.append(error_calculate(3,true_center,res))
    index.append(i)
plot_graph()
