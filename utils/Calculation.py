import numpy as np
import random

def GetAllTheta(Data, feature):
	Data = Data[:,feature]
	Data = sorted(Data)
	AllTheta = []
	for i in range(1, len(Data)):
		if Data[i] == Data[i-1]:
			continue
		AllTheta.append((Data[i] + Data[i-1])/2)
	return AllTheta

def GiniImpurity(Data, feature, theta):
	TotalLen = len(Data)

	# Sort Data with feature
	SortedData = sorted(Data, key = lambda x: x[feature])

	# Find cut index
	Cut = -1
	for i in range(TotalLen):
		if SortedData[i][feature] > theta:
			Cut = i
			break

	# Cut data to left & right
	Left, Right = SortedData[:Cut], SortedData[Cut:]
	LeftLen, RightLen = len(Left), len(Right)
	LeftLabel, RightLabel = -1, 1

	# Calculate gini impurity
	# Left:
	Count = 0
	for i in Left:
		if i[-1] * LeftLabel < 0:
			Count += 1
	Error = Count / LeftLen
	LeftGini = 1 - Error**2 - (1-Error)**2

	# Right:
	Count = 0
	for i in Right:
		if i[-1] * RightLabel < 0:
			Count += 1
	Error = Count / RightLen
	RightGini = 1 - Error**2 - (1-Error)**2

	# Sum up:
	Gini = (LeftLen / TotalLen * LeftGini) + (RightLen / TotalLen * RightGini)

	return Gini

def CheckLeaf(Data):
	X = Data[:,:-1]
	Y = Data[:,-1]

	Xsame = np.sum(X != X[0])
	Ysame = np.sum(Y != Y[0])

	return (Xsame == 0) or (Ysame == 0)

def sign(x):
	return 1 if x >= 0 else -1

def ChooseLabel(Data):
	return sign(sum(Data))

def TreePredict(Tree, Sample):
	while not Tree.isleaf:
		# print("X[%d] >= %f ?" % (Tree.feature, Tree.theta), end = "\t")
		if (Sample[Tree.feature] - Tree.theta) < 0:
			# print("no, go left")
			Tree = Tree.left
		else:
			# print("yes, go right")
			Tree = Tree.right
	# print("At leaf, predict %d" % (Tree.label))
	return Tree.label

def TreePrediction(Tree, Data):
	Prediction = np.array([TreePredict(Tree, x) for x in Data])
	return Prediction

def CalculateError(Data, Prediction):
	Y = Data[:,-1]
	Error = np.mean(Y*Prediction < 0)
	return Error

def ForestPrediction(Forest, Data):
	Predictions = []
	for Tree in Forest:
		Prediction = TreePrediction(Tree, Data)
		Predictions.append(Prediction)
	return np.array(Predictions)

def UniformBlending(Predictions):
	Prediction = np.sum(Predictions, axis = 0)
	Prediction = np.array([sign(x) for x in Prediction])
	return Prediction