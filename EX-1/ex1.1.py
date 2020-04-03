import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv('ex1data1.txt', header = None, sep=",")
X = data.iloc[:,0] 
y = data.iloc[:,1]
m = len(y) 

X = X[:, np.newaxis]
y = y[:, np.newaxis]
theta = np.zeros([2,1])
iterations = 10000
alpha = 0.01
ones = np.ones([m,1])
X = np.hstack((ones,X))

def computeCost(X, y, theta):
	temp = np.dot(X, theta) - y
	return np.sum(np.power(temp,2))/(2*m)

def gradientDescent(X, y, theta, alpha, iterations):
	for i in range(iterations):
		temp = np.dot(X, theta) - y
		temp = np.dot(X.T, temp)
		theta = theta - ((alpha/m)*temp)

	return theta
	
theta = gradientDescent(X, y, theta, alpha, iterations)		
print(theta)

J = computeCost(X, y, theta)
print(J)

plt.scatter(X[:,1], y, marker = "x", color= "red")
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta), color = "blue")
#plt.savefig('graph_1.png')

ppl = int(input("Enter the population in 10,000s: "))
ppl_arr = [1,ppl]

profit = np.dot(ppl_arr, theta)
print("Expected profit: ")
print(profit)


plt.show()














       # FROMN INBUILT PYTHON LIBRARY
# data = pd.read_csv('ex1data1.txt', delimiter=',')  # load data set
# X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
# Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions

# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')
# print()

# plt.scatter(X, Y, marker="x")
# plt.plot(X, Y_pred, color='red')
# plt.show()