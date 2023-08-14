from __future__ import print_function
import numpy as np
from sklearn import cluster
from sklearn.model_selection import train_test_split
from model import KMeans

np.random.seed(18)
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# My model
model = KMeans(n_clusters= 3, max_iter= 1000)
model.fit(X)
print("\nCenters found by my algorithm:\n", model.cluster_centers_)



# Using SKlearn's library
model = cluster.KMeans(n_clusters= 3, max_iter= 1000, n_init= 'auto')
model.fit(X)
print("\nCenters found by scikit-learn:\n", model.cluster_centers_)