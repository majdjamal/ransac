
"""ransac.py: Random sample consensus used to remove outliers from a dataset."""

__author__ = "Majd Jamal"

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class RANSAC:
	"""
	Random Sample Concensus
	"""
	def __init__(self, iterations = 100, residual_threshold = 1):

		self.iterations = iterations
		self.residual_threshold = residual_threshold

		self.optimal_slope = None
		self.optimal_intercept = None
		self.inliers = None

	def func(self, x, slope, intercept):
		""" Linear function
		:param x: x-coordinates
		:param slope: slope value
		:param intercept: intercept value
		:return: y-coordinates
		"""
		return slope*x + intercept

	def getParams(self):
		""" Returns RANSAC parameters
		"""
		return self.optimal_slope, self.optimal_intercept

	def getInliers(self):
		""" Returns dataset without outliers
		"""
		return self.inliers

	def fit(self, X):
		""" Train the RANSAC regressor. This function finds
			the optimal parameters and inliers.
		:param X: Data points, with shape (Npts x Ndim = 2)
		"""
		Npts, _ = X.shape

		x1 = X[:, 0]
		x2 = X[:, 1]

		Ninliers = 0

		for itr in range(self.iterations):

			rndPoints = np.random.randint(low = 0, high=Npts, size=2)
			rndPoints = X[rndPoints, :]

			slope, intercept, _, _, _ = stats.linregress(rndPoints[:, 0], rndPoints[:,1])
			y_pred = self.func(x1, slope, intercept)
			residual_error = np.subtract(y_pred, x2)
			indices = np.where(np.abs(residual_error) < self.residual_threshold )

			if indices[0].size > Ninliers:

				self.optimal_slope = slope
				self.optimal_intercept = intercept

				self.inliers = X[indices[0]]
				Ninliers = indices[0].size

	def predict(self, x):
		""" Predicts if a point is an inlier or outlier.
		:param x: a data point.
		:return: a signal, 1 means inlier and 0 means outlier.
		"""
		x1 = x[0]
		x2 = x[1]

		slope = self.optimal_slope
		intercept = self.optimal_intercept

		y_pred = self.func(x1, slope, intercept)

		residual_error = np.subtract(x2, y_pred)

		if abs(residual_error) < self.residual_threshold:
			return 1
		else:
			return 0
