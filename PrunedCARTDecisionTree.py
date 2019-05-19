import numpy as np
import sys
import random
from utils.ReadData import *
from utils.Tree import *
from utils.Calculation import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MAXHEIGHT = 10

def CART(Data, H):
	# Check if maximum hight is reached
	if H <= 0:
		LeafNode = TreeNode()
		LeafNode.isleaf = True
		LeafNode.label = ChooseLabel(Data[:,-1])
		return LeafNode

	# Check if all data are the same label or all xi are the same
	if CheckLeaf(Data):
		LeafNode = TreeNode()
		LeafNode.isleaf = True
		LeafNode.label = int(Data[0][-1])
		return LeafNode

	# Try all possible cuts
	Dim = len(Data[0]) - 1
	TrackGini = []
	for feature in range(Dim):
		AllTheta = GetAllTheta(Data, feature)
		for theta in AllTheta:
			Gini = GiniImpurity(Data, feature, theta)
			TrackGini.append((Gini, feature, theta))
	
	# Choose best cut with minimun Gini impurity
	SortedGini = sorted(TrackGini, key = lambda x: x[0])
	BestGini = SortedGini[0] # [Gini, feature, theta]
	_, feature, theta = BestGini
	root = TreeNode(feature, theta)
	
	# Find left & right index
	LeftID = Data[:, feature] < theta
	RightID = Data[:, feature] >= theta

	# Cut data to left & right
	LeftData, RightData = Data[LeftID], Data[RightID]

	# Generate subtree
	root.left = CART(LeftData, H-1)
	root.right = CART(RightData, H-1)

	# return root
	return root


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python CARTDecisionTree.py TrainData TestData")
		exit(0)

	# Read in data
	TrainFile, TestFile = sys.argv[1:3]
	TrainData = ReadData(TrainFile)
	TestData = ReadData(TestFile)

	# Experiment with tree hight
	TrackEin, TrackEout = [], []
	for Height in range(MAXHEIGHT):
		print("Height:", Height, end = "\t")
		# Train CART decision tree
		CARTDTree = CART(TrainData, Height)

		# Predict train data
		Prediction = TreePrediction(CARTDTree, TrainData)
		Ein = CalculateError(TrainData, Prediction)
		print("Ein:", Ein, end = "\t")
		TrackEin.append(Ein)

		# Predict test data
		Prediction = TreePrediction(CARTDTree, TestData)
		Eout = CalculateError(TestData, Prediction)
		print("Eout:", Eout)
		TrackEout.append(Eout)

	# Plot graph of H v.s. Ein and H v.s. Eout
	X = list(range(MAXHEIGHT))
	plt.figure("Problem 13")
	plt.title("$Error(g_h)\ v.s.\ Height$")
	Ein_patch = mpatches.Patch(color='blue', label='$E_{in}(g_h)$')
	Eout_patch = mpatches.Patch(color='red', label='$E_{out}(g_h)$')
	plt.legend(handles=[Ein_patch, Eout_patch])
	plt.plot(X, TrackEin, 'b-', X, TrackEout, 'r-')
	plt.show()