import numpy as np
from time import time
from scipy.cluster.vq import *
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 20,10
animal_list = []
data_list = []

with open ('classes.txt') as File:
    for line in File:
        animal_list.append(line.split()[1])

with open ('predicate-matrix-continuous.txt') as File:
    for line in File:
        #data_list.append(np.array(map(float, line.split())))
        data_list.append(map(float, line.split()))
#data_list = np.array(data_list)
#print data_list
#res, idx = kmeans2(data_list, 3)
kmean = KMeans(10, 'k-means++', 100, 1)
t0 = time()
kmean.fit(data_list)
print("done in %0.3fs" % (time() - t0))

plt.title("Hierarchy linkage")
dendrogram(linkage(data_list, method='ward'),
        color_threshold=1,labels=animal_list,orientation='right',show_leaf_counts=True)
tuple_list = []
for i in range(50):
    tuple_list.append((animal_list[i], kmean.labels_[i]))

tuple_list.sort(key=lambda t: t[1])
print tuple_list
plt.show()
