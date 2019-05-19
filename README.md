# CART decision tree and random forest
Practice of CART decision tree and random forest.

## How to run

Start the program
```
python CARTDecisionTree.py TrainData TestData
```
```
python PrunedCARTDecisionTree.py TrainData TestData
```
```
python RandomForest.py TrainData TestData
```

## Description

### CARTDecisionTree.py

Implementing the CART decision tree, using Gini impurity as the branching factor.

### PrunedCARTDecisionTree.py

Pruned version of the CART decision tree, using a simple pruning technique: simply restrict the maximum height.

### RandomForest.py

Aggregate 30000 fully grown CART decision trees, trained with bagging 0.8 of the training data.

## Built With

* Python 3.6.0 :: Anaconda custom (64-bit)

## Authors

* **SaKaTetsu** - *Initial work* - [SaKaTetsu](https://github.com/SaKaTetsu)