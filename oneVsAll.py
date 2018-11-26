import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid

def oneVsAll(X, y, num_labels, lambda_t):
	m, n = X.shape
	all_theta = np.zeros((num_labels, n + 1))
	X = np.hstack((np.ones((m,1)),X))
	
	for i in range (0, num_labels):
		initial_theta = np.zeros((n + 1, 1))
		res = minimize(cost_func_reg, initial_theta, method='BFGS', jac=gradient_func_reg, args=(X,(y==i), lambda_t), options={'disp': True, 'maxiter':400})
			  
			  
def cost_func_reg(theta, X, y, l):
    theta = np.asarray(np.matrix(theta).T)
    mask_array = np.ones((len(theta), 1))
    mask_array[0,0] = 0.0
    
    with np.errstate(divide='ignore'):
        m = len(y)  
        htheta = sigmoid(np.matmul(X, theta))
        # use auxiliary gradient function
        # use auxiliary cost function
        J = - 1 / m * (np.matmul(y.T, np.log(htheta)) + np.matmul((1-y).T, np.log(1-htheta))) \
                + l/2/m*np.sum(np.multiply(mask_array, theta)**2)
        return J

# Take the following parameters and return the cost and gradient
# theta = 1 X m, X = n X m, and y = 1 X n (m = no. of feature and n = no. of sample)
# grad = 1 X m
def gradient_func_reg(theta, X, y, l):
    theta = np.asarray(np.matrix(theta).T)
    mask_array = np.ones((len(theta), 1))
    mask_array[0, 0] = 0.0
    
    with np.errstate(divide='ignore'):
        m = len(y)  
        htheta = sigmoid(np.matmul(X, theta))
        
        grad = np.matmul(X.T, (htheta - y)) / m + np.multiply(mask_array, theta) * l / m
        grad = grad.ravel()
        return grad			  