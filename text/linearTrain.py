################################################################################
#
#	Train a decision tree from data
#
#	Ammon Dodson
#
################################################################################


import sys

sys.path.insert(0,'../')
import dataPreProcess as dpp

import pandas
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import KFold
from sklearn                 import metrics
from joblib                  import dump

#classes = ['age','ope','con','ext','agr','neu']

############################## COLLECT SAMPLE DATA #############################

dataPath = sys.argv[1]
if not dpp.verify(dataPath):
	print("could not find data")
	exit()

trainingData = dpp.combineLIWC(dpp.loadProfile(dataPath), dpp.loadLIWC(dataPath))
#print(trainingData)

X = trainingData[dpp.LIWC]
y = trainingData[dpp.CLASSES]

#print(X)
#print(y)

model = LinearRegression()

##################################### TEST #####################################

folds = KFold(10,True)

for classification in dpp.CLASSES:
	mean = 0.0
	
	print(classification)
	
	#print(y[classification])
	
	for train_index, test_index in folds.split(X):
		Xtrain = X.iloc[train_index]
		Xtest  = X.iloc[test_index]
		yTrain = y[classification].iloc[train_index]
		yTest  = y[classification].iloc[test_index]
	
		model.fit(Xtrain, yTrain)
	
		accuracy = metrics.mean_squared_error(yTest,model.predict(Xtest))
		print(accuracy)
		mean += accuracy
	
	mean /= 10
	print(classification + " mean: " + str(mean))
	

#################################### TRAIN #####################################

model.fit(X, y)

################################## SAVE MODEL ##################################

dump(model, "linearRegression.joblib")


