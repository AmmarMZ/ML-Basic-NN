
from scipy.io import loadmat
import numpy as np

from displayData import displayData


data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']


# m := 5000, m is the number of rows in the data
m = len(X);

# rand_indices is a 5000 x 1 matrix with values between 0-5000 (or 1-5000)
rand_indices = np.random.permutation(m)


# get 100 random indices and then get the row of x that corresponds to that values
# put 100 of those rows in @sel
# recall each row of X represents a 20x20 image of a number "rolled out"
sel = X[rand_indices[1:100],:];  

displayData(sel);

