import sys
import numpy as np
import random
from CARTDecisionTree import *
from utils.Calculation import *
from utils.ReadData import *
import matplotlib.pyplot as plt

NumofTree = 30000
BagSize = 0.8

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python RandomForest.py TrainData TestData")
		exit(0)

	# Read in data
	TrainFile, TestFile = sys.argv[1:3]
	TrainData = ReadData(TrainFile)
	TestData = ReadData(TestFile)

	# Generate forest
	Forest = []
	for i in range(NumofTree):
		# Bagging
		BagData = Bagging(TrainData, BagSize)

		# Generate Tree
		CARTDTree = CART(BagData)

		# Add tree into forest
		Forest.append(CARTDTree)

		# Print every 100 trees
		if i % 100 == 99:
			print("Generated %d trees" % (i+1))

	# Prediction
	TrainPredictions = ForestPrediction(Forest, TrainData)
	TestPredictions = ForestPrediction(Forest, TestData)

	# Problem 14
	# Ein(gt) over the 30000 trees
	TrackEingt = []
	for Prediction in TrainPredictions:
		TrackEingt.append(CalculateError(TrainData, Prediction))

	# Plot histogram
	plt.figure("Problem 14")
	plt.title("$E_{in}(g_t)$")
	plt.hist(TrackEingt)

	# Problem 15
	# Ein(Gt) v.s. t
	TrackEinGt = []
	for i in range(1, NumofTree+1):
		Prediction = UniformBlending(TrainPredictions[:i])
		TrackEinGt.append(CalculateError(TrainData, Prediction))

	# Plot curve
	t = list(range(1, NumofTree+1))
	plt.figure("Problem 15")
	plt.title("$E_{in}(G_t)\ v.s.\ t$")
	plt.plot(t, TrackEinGt)

	print("Ein:", TrackEinGt[-1])

	# Problem 16
	# Eout(Gt) v.s. t
	TrackEoutGt = []
	for i in range(1, NumofTree+1):
		Prediction = UniformBlending(TestPredictions[:i])
		TrackEoutGt.append(CalculateError(TestData, Prediction))

	# Plot curve
	plt.figure("Problem 16")
	plt.title("$E_{out}(G_t)\ v.s.\ t$")
	plt.plot(t, TrackEoutGt)

	print("Eout:", TrackEoutGt[-1])

	plt.show()