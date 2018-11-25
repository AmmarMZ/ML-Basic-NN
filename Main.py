from scipy.io import loadmat
import numpy as np
from lrCostFunction import lrCostFunction

from displayData import displayData

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

# m := 5000, m is the number of rows in the data
m = len(X)

# rand_indices is a 5000 x 1 matrix with values between 0-5000 (or 1-5000)
rand_indices = np.random.permutation(m)


# get 100 random indices and then get the row of x that corresponds to that values
# put 100 of those rows in @sel
# recall each row of X represents a 20x20 image of a number "rolled out"
sel = X[rand_indices[1:100],:];  

displayData(sel)

theta_t = np.array([[-2],[-1],[1],[2]])


# X_t is a 5 x 4 array where a column of 1's is added on to the left of a 5 x 3 matrix 
X_t = np.hstack([np.ones((5,1)), np.reshape(np.arange(1, 16),(5,3), order='F')/10])

y_t = np.array([[1],[0],[1],[0],[1]])
lambda_t = 3

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)



print('\nCost: %f\n' % J);
print('Expected cost: 2.534819\n');
print('Gradients:\n');
print(grad);
print('Expected gradients:\n');
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');