#######
#
#	Survey learning models
#
#######


import sys

sys.path.insert(0,'../')
import utility

dataPath = sys.argv[1]
if not utility.verify(dataPath):
	print("could not find data")
	exit()

trainingData = utility.combineLIWC(
	utility.loadProfile(dataPath),
	utility.loadLIWC(dataPath)
)
#print(trainingData)

X = trainingData[utility.LIWC]


################################### BASELINE ###################################

#for c,v in [dpp.Y,[0.59,0.59,0.65,0.80,0.79,0.66,0.73]]:
#	print(c + ": " + str(v))

################################# DECISION TREE ################################

print('\nDECISION TREE\n')
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=6, criterion='entropy')
print(model.get_params())

utility.kFoldCrossAccuracy(model, X, trainingData['gender'], 10)

#model.fit(X,trainingData['gender'])
#print(len(model.feature_importances_))
#print(len(utility.LIWC))
##print(model.feature_importances_)

#for imp,cl in zip(model.feature_importances_,utility.LIWC):
#	if imp > 0:
#		print(cl + ": " + str(imp))

#print(model.tree_)

################################# NAIVE BAYES ##################################

print('\nGAUSSIAN NAIVE BAYES\n')
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
print(model.get_params())

utility.kFoldCrossAccuracy(model, X, trainingData['gender'], 10)

print('\nMULTINOMIAL NAIVE BAYES\n')
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
print(model.get_params())

utility.kFoldCrossAccuracy(model, X, trainingData['gender'], 10)

print('\nCOMPLEMENT NAIVE BAYES\n')
from sklearn.naive_bayes import ComplementNB

model = ComplementNB(norm=True)
print(model.get_params())

utility.kFoldCrossAccuracy(model, X, trainingData['gender'], 10)

############################## LINEAR REGRESSION ###############################

print("\nLINEAR REGRESSION\n")
from sklearn.linear_model    import LinearRegression

model = LinearRegression() # support vector regression
print(model.get_params())

for classification in utility.Y_NUMERICAL:
	print("  == " + classification + " ==")
	utility.kFoldCrossRMSE(model, X, trainingData[classification], 10)

############################ SUPPORT VECTOR MACHINES ###########################

print("\nSUPPORT VECTOR MACHINES\n")
from sklearn import svm

print('=== gender ===')
model = svm.SVC(gamma='scale')
print(model.get_params())

utility.kFoldCrossAccuracy(model, X, trainingData['gender'], 3)


model = svm.SVR(gamma='scale') # support vector regression
print(model.get_params())

for classification in utility.Y_NUMERICAL:
	print("  == " + classification + " ==")
	utility.kFoldCrossRMSE(model, X, trainingData[classification], 3)

