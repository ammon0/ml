################################################################################
#
#	Hyperparameter testing to determine the best depth of decision tree for
#	gender
#
#	Results: best depth: 6 with accuracy: 0.631742862098222
#
#	Ammon Dodson
#
################################################################################


import pandas
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


dataTable = pandas.read_csv("liwcTrain.csv", index_col=0)

dataTable.drop('userId', axis=1, inplace=True)

#print(dataTable)

features = list(dataTable.columns[:82])
#classes  = list(dataTable.columns[82:])

#print("features: " + str(features))
#print("classes: " + str(classes))


#################################### GENDER ####################################

X      = dataTable[features]
gender = dataTable["gender"]

bestDepth = 0
bestAccuracy = 0.0

for i in range(1,40):
	mean = 0.0
	
	model = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy')


	# 10-fold
	skf = StratifiedKFold(n_splits=10)
	skf.get_n_splits(X,gender)
	for train_index, test_index in skf.split(X,gender):
		Xtrain, Xtest = X     .iloc[train_index], X     .iloc[test_index]
		yTrain, yTest = gender.iloc[train_index], gender.iloc[test_index]
	
		model.fit(Xtrain, yTrain)
	
		accuracy = metrics.accuracy_score(yTest,model.predict(Xtest))
		print(accuracy)
		mean += accuracy

	mean /= 10
	print(str(i) + ": mean: " + str(mean))
	if mean > bestAccuracy:
		bestDepth = i
		bestAccuracy = mean
	elif mean+0.02 < bestAccuracy: # starting to decline
		break

print("best depth: " + str(bestDepth) + " with accuracy: " + str(bestAccuracy))

