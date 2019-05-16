################################################################################
#
#	Train a decision tree from data
#
#	Ammon Dodson
#
################################################################################


import sys

sys.path.insert(0,'../')
import utility

import pandas
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import KFold
from sklearn                 import metrics
from joblib                  import dump
from numpy                   import sqrt

#classes = ['age','ope','con','ext','agr','neu']

############################## COLLECT SAMPLE DATA #############################

dataPath = sys.argv[1]
if not utility.verify(dataPath):
	print("could not find data")
	exit()

trainingData = utility.combineLIWC(utility.loadProfile(dataPath), utility.loadLIWC(dataPath))
#print(trainingData)

X = trainingData[utility.LIWC]
y = trainingData[utility.Y_NUMERICAL]

#print(X)
#print(y)

model = LinearRegression()

##################################### TEST #####################################

folds = KFold(10,True)

for classification in utility.Y_NUMERICAL:
	mean = 0.0
	
	#print(classification)
	
	#print(y[classification])
	
	for train_index, test_index in folds.split(X):
		Xtrain = X.iloc[train_index]
		Xtest  = X.iloc[test_index]
		yTrain = y[classification].iloc[train_index]
		yTest  = y[classification].iloc[test_index]
	
		model.fit(Xtrain, yTrain)
	
		accuracy = sqrt(metrics.mean_squared_error(yTest,model.predict(Xtest)))
		#print('RMSE: ' + str(accuracy))
		mean += accuracy
	
	mean /= 10
	print(classification + " mean RMSE: " + str(mean))
	

#################################### TRAIN #####################################

model.fit(X, y)

#print(list(zip(model.coef_, utility.CLASSES)))

################################## SAVE MODEL ##################################

dump(model, "linearRegression.joblib")


