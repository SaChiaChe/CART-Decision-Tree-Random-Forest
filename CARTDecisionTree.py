import numpy as np
import sys
from utils.ReadData import *
from utils.Tree import *
from utils.Calculation import *

def CART(Data):
	# Check if all data are the same label
	if CheckLeaf(Data):
		LeafNode = TreeNode()
		LeafNode.isleaf = True
		LeafNode.label = Data[0][-1]
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
	root.left = CART(LeftData)
	root.right = CART(RightData)

	# return root
	return root

def PrintTree(root):
	if root is None:
		return

	print(root.feature, root.theta, root.isleaf, root.label)

	PrintTree(root.left)
	PrintTree(root.right)


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python CARTDecisionTree.py TrainData TestData")
		exit(0)

	# Read in data
	TrainFile, TestFile = sys.argv[1:3]
	TrainData = ReadData(TrainFile)
	TestData = ReadData(TestFile)

	# Train CART decision tree
	CARTDTree = CART(TrainData)

	# Predict train data
	Prediction = TreePrediction(CARTDTree, TrainData)
	Ein = CalculateError(TrainData, Prediction)
	print("Ein:", Ein)

	# Predict test data
	Prediction = TreePrediction(CARTDTree, TestData)
	Eout = CalculateError(TestData, Prediction)
	print("Eout:", Eout)

	# Print tree
	# PrintTree(CARTDTree)