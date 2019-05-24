################################################################################
#
#	Survey learning models
#
################################################################################


import sys

sys.path.insert(0,'../')
import utility

dataPath = sys.argv[1]
if not utility.verify(dataPath):
	print("could not find data")
	exit()

pt = utility.loadProfile(dataPath)
pt = utility.ageCategorize(pt)
#print(pt)

trainingData = utility.combineLIWC(pt, utility.loadLIWC(dataPath))
#print(trainingData)

X = trainingData[utility.LIWC]


SLOW_FOLDS = 3
FAST_FOLDS = 10

################################### BASELINE ###################################

#for c,v in [dpp.Y,[0.59,0.59,0.65,0.80,0.79,0.66,0.73]]:
#	print(c + ": " + str(v))

############################## LOGISTIC REGRESSION #############################

print("\nLOGICISTIC REGRESSION\n")
from sklearn.linear_model   import LogisticRegression

model = LogisticRegression(
	solver='liblinear',
	max_iter=100
)
print(model.get_params())

print("  == gender ==")
utility.kFoldCrossAccuracy(model, X, trainingData['gender'], FAST_FOLDS)


model = LogisticRegression(
	solver="lbfgs",
	multi_class="auto",
	max_iter=10000
)
print(model.get_params())

print("  == ageCat ==")
utility.kFoldCrossAccuracy(model, X, trainingData['ageCat'], SLOW_FOLDS)

################################# DECISION TREE ################################

print('\nDECISION TREE\n')
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=6, criterion='entropy')
print(model.get_params())

for cl in utility.Y_CATEGORICAL:
	print("  == " + cl + " ==")
	utility.kFoldCrossAccuracy(model, X, trainingData[cl], FAST_FOLDS)

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

for cl in utility.Y_CATEGORICAL:
	print("  == " + cl + " ==")
	utility.kFoldCrossAccuracy(model, X, trainingData[cl], FAST_FOLDS)

print('\nMULTINOMIAL NAIVE BAYES\n')
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
print(model.get_params())

for cl in utility.Y_CATEGORICAL:
	print("  == " + cl + " ==")
	utility.kFoldCrossAccuracy(model, X, trainingData[cl], FAST_FOLDS)

print('\nCOMPLEMENT NAIVE BAYES\n')
from sklearn.naive_bayes import ComplementNB

model = ComplementNB(norm=True)
print(model.get_params())

for cl in utility.Y_CATEGORICAL:
	print("  == " + cl + " ==")
	utility.kFoldCrossAccuracy(model, X, trainingData[cl], FAST_FOLDS)

############################## LINEAR REGRESSION ###############################

print("\nLINEAR REGRESSION\n")
from sklearn.linear_model    import LinearRegression

model = LinearRegression() # support vector regression
print(model.get_params())

for cl in utility.Y_NUMERICAL:
	print("  == " + cl + " ==")
	utility.kFoldCrossRMSE(model, X, trainingData[cl], FAST_FOLDS)

############################ SUPPORT VECTOR MACHINES ###########################

print("\nSUPPORT VECTOR MACHINES\n")
from sklearn import svm

print('=== gender ===')
model = svm.SVC(gamma='scale') # categorical
print(model.get_params())

for cl in utility.Y_CATEGORICAL:
	print("  == " + cl + " ==")
	utility.kFoldCrossAccuracy(model, X, trainingData[cl], SLOW_FOLDS)

#utility.kFoldCrossAccuracy(model, X, trainingData['gender'], 3)


model = svm.SVR(gamma='scale') # support vector regression
print(model.get_params())

for cl in utility.Y_NUMERICAL:
	print("  == " + cl + " ==")
	utility.kFoldCrossRMSE(model, X, trainingData[cl], SLOW_FOLDS)

