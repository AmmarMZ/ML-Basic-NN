from scipy.io import loadmat
import numpy as np

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']


# m := 5000, m is the number of rows in the data
m = len(X);

rand_indices = np.random.permutation(m)
sel = X[rand_indices[1:100],:]