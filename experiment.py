
import numpy as np
from ransac import RANSAC
import matplotlib.pyplot as plt

def func(x):
	return 2*x + 2

##
##	Generate data
##
x = np.linspace(0, 5, 80)
y = func(x) + np.random.randn(80) * 0.9 + 1

x = np.concatenate((x, np.random.randn(20) * 1 +  5)) 	#Add outliers
y = np.concatenate((y, np.random.randn(20) * 1 +  5))	#Add outliers

X = np.vstack((x, y))
X = X.T

##
##	Testing RANSAC
##
rns = RANSAC(residual_threshold = 2.5, iterations = 300)
rns.fit(X)

inliers = rns.getInliers()

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(inliers[:, 0], inliers[:, 1], color='red')
plt.show()
