import matplotlib.pyplot as plt
import numpy as np
#mean = [0,0]
#cov = [[1,0],[0,100]]
mean = [1,2,3] #mean distribution
cov = [[2,0,0],[0,2,0],[0,0,2]] #covariance matrix
x,y,z = np.random.multivariate_normal(mean,cov,10).T

print np.random.multivariate_normal(mean,cov,10).T
#print zip(x,y)
plt.plot(x,y,'x')
plt.axis('equal')
plt.show()
