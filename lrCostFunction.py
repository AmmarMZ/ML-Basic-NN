from sigmoid import sigmoid
import numpy as np


def lrCostFunction(theta,X,y,l):
	m = len(y)
	J = 0
	grad = np.zeros((len(theta),len(theta[0])))
	tempTheta = np.ones(1)
	tempTheta = theta;
	tempTheta[0] = 0
	sTheta = sigmoid(np.matmul(X,theta))
	
	s1 = y.T
	s2 = np.log(sTheta)
	
	m1 = np.matmul(s1, s2)
	
	s3 = (1-y).T
	s4 = np.log(1-sTheta)
	
	m2 =  np.matmul(s3, s4)
	
	J = (-1/m) * np.sum(m1 + m2)
	temp = sigmoid(np.matmul(X,theta))
	error = temp - y
	
	grad = (1/m) * (np.matmul(X.T,error)) + (l/m)*tempTheta
	return J, grad
	


