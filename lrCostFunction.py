from sigmoid import sigmoid


def lrCostFunction(theta,X,y,l):
	m = len(y)
	J = 0
	grad = np.zeros((len(theta),len(theta[0]))
	tempTheta = np.ones(1)
	#np.copy(tempTheta,theta)
	tempTheta(0) = 0
	
	sTheta = sigmoid(np.matmul(X,theta))
	J = (-1/m) * np.sum(np.matmul(y.T , log(sTheta)) + np.matmul((1-y).T, np.log(1-sTheta)))
	temp = sigmoid(matmul(X,theta))
	error = temp - y
	grad = (1/m) * (np.matmul(X.t,error)) + (l/m)*tempTheta
	return J, grad