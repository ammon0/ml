################################################################################
#
#	Train age and gender with linear regression
#
#	By: Ammon Dodson
#
################################################################################



import sys

sys.path.insert(0,'../')
import utility

from sklearn.linear_model import LogisticRegression
from joblib               import dump

############################## COLLECT SAMPLE DATA #############################

dataPath = sys.argv[1]
if not utility.verify(dataPath):
	print("could not find data")
	exit()

trainingData = utility.combineLIWC(
	utility.loadProfile(dataPath),
	utility.loadLIWC(dataPath)
)
trainingData = utility.ageCategorize(trainingData)
#print(trainingData)


X = trainingData[utility.LIWC]

genderY = trainingData['gender']
ageY    = trainingData['age']


#print(X)
#print(ageY)

genderModel = LogisticRegression(
	solver='liblinear',
	max_iter=100
)

ageModel = LogisticRegression(
	solver="lbfgs",
	multi_class="auto",
	max_iter=10000
)

##################################### TEST #####################################

print("  == gender ==")
utility.kFoldCrossAccuracy(genderModel, X, trainingData['gender'], 10)

print("  == ageCat ==")
utility.kFoldCrossAccuracy(ageModel, X, trainingData['age'], 3)


#################################### TRAIN #####################################

genderModel.fit(X, genderY)
ageModel   .fit(X, ageY)

################################## SAVE MODEL ##################################

dump(genderModel, "genderLogReg.joblib")
dump(ageModel   , "ageLogReg.joblib")




