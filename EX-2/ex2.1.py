import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('ex2data1.txt', delimiter=',', header = None)

res = df.iloc[:,2].values
exam_1 = df.iloc[:,0].values
exam_2 = df.iloc[:,1].values

# X = feature values, all the columns except the last column
X = df.iloc[:, :-1]

# y = target values, last column of the data frame
y = df.iloc[:, -1]

ex1_pos = []
ex2_pos = []
ex1_neg = []
ex2_neg = []

for i in range(100):
	if(res[i]==1):
		ex1_pos.append(exam_1[i])
		ex2_pos.append(exam_2[i])
	else:
		ex1_neg.append(exam_1[i])
		ex2_neg.append(exam_2[i])

plt.xlabel('Exam-1 score')
plt.ylabel('Exam-2 score')

plt.scatter(ex1_neg,ex2_neg, marker="o", label="Not admiited")
plt.scatter(ex1_pos,ex2_pos, marker = "+", label= "Admiited")
plt.legend()

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.plot(X, y_pred, color='red')

plt.show()
