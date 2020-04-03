import numpy as np
import pandas as pd

data = pd.read_csv('ex1data2.txt', header = None, sep=",")

X = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)

X = (X-np.mean(X))/np.std(X)

y = y[:, np.newaxis]
theta = np.zeros([3,1])
ones = np.ones([m,1])
X = np.hstack((ones,X))
iterations = 400
alpha = 0.01

def computeCost(X, y, theta):
	temp = np.dot(X, theta) - y
	return np.sum(np.power(temp, 2))/(2*m)

def gradientDescent(X, y, theta, alpha, iterations):
	for i in range(iterations):
		temp = np.dot(X, theta) - y
		temp = np.dot(X.T, temp)
		theta= theta - ((alpha/m)*temp)
	return theta

theta = gradientDescent(X, y, theta, alpha, iterations)

J = computeCost(X, y, theta)

print("Cost = "+str(J))
