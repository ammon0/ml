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
from sklearn import tree
from sklearn import metrics
from joblib import dump, load

dataPath = sys.argv[1]
if not utility.verify(dataPath):
	print('could not find data')
	exit()

trainingData = utility.combineLIWC(
	utility.loadProfile(dataPath),
	utility.loadText(dataPath)
)

#trainingData = pandas.read_csv("liwcTrain.csv", index_col=0)
#trainingData.drop('userId', axis=1, inplace=True)

#features = list(trainingData.columns[:82])
#X      = trainingData[features]
#gender = trainingData["gender"]

X      = trainingData[utility.LIWC]
gender = trainingData['gender']

#################################### TRAIN #####################################

model = tree.DecisionTreeClassifier(max_depth=6, criterion='entropy')

model.fit(X, gender)

################################## SAVE MODEL ##################################

dump(model, "genderTree.joblib")

##################################### TEST #####################################

#testData = pandas.read_csv('liwcTest.csv', index_col=0)

#X      = trainingData[features]
#gender = trainingData["gender"]

#accuracy = metrics.accuracy_score(gender,model.predict(X))
#print(accuracy)
