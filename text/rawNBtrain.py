################################################################################
#
#	Raw text training
#
################################################################################


from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from numpy import sqrt

import time

import sys
sys.path.insert(0,'../')
import utility
sys.path.insert(0, './text/')
import textPreProcess as tpp

startTime = time.time()


################################################################################
#                                 COLLECT DATA
################################################################################


dataPath = sys.argv[1]
if not utility.verify(dataPath):
	print("could not find data")
	exit()

print('loading data')
pt = utility.loadProfile(dataPath)
pt = utility.ageCategorize(pt)

trainingData = utility.combineLIWC(pt, utility.loadText(dataPath))
print('data loaded {:.1f}s'.format(time.time() - startTime))


################################################################################
#                              TEXT PREPROCESSING
################################################################################


print('preprocessing')

trainingData = tpp.splitStatusUpdates(trainingData) #99655 features

trainingData['status'] = trainingData['status'].apply( # 99655 features
	lambda t: t.lower())
trainingData['status'] = trainingData['status'].apply( # 98254 features
	lambda t: tpp.removeURL(t))
trainingData['status'] = trainingData['status'].apply( # 95171 features
	lambda t: tpp.reduceRepetition(t))
trainingData['status'] = trainingData['status'].apply( # 94971 features
	lambda t: tpp.replaceEmojies(t))
trainingData['status'] = trainingData['status'].apply( # 94822 features
	lambda t: tpp.laughReduction(t))
trainingData['status'] = trainingData['status'].apply( # 94748 features
	lambda t: tpp.normalizeMoney(t))
trainingData['status'] = trainingData['status'].apply( # 94748 features
	lambda t: tpp.normalize(t))
trainingData['status'] = trainingData['status'].apply( # 91392 features
	lambda t: tpp.noiseRemoval(t))
trainingData['status'] = trainingData['status'].apply( # 72581 features
	lambda t: tpp.stemmer(t))

print('preprocess complete {:.1f}s'.format(time.time() - startTime))


################################################################################
#                               COUNT VECTORIZER
################################################################################


cv = CountVectorizer(ngram_range=(1, 3),min_df=3)
#X = cv.fit_transform(trainingData['status'])
cv.fit(trainingData['status'])
print(len(cv.get_feature_names()))
#print(cv.get_feature_names())

dump(cv, "countVector.joblib")


print('vectorized {:.1f}s'.format(time.time() - startTime))


################################################################################
#                                 EVALUATION
################################################################################


model = MultinomialNB()

#for cl in utility.Y_CATEGORICAL:
#	print("  == " + cl + " ==")
#	utility.kFoldCrossAccuracy(model, X, trainingData[cl], 10)


################################################################################
#                                 TRAINING
################################################################################


X = cv.transform(trainingData['status'])

for cl in utility.Y_CATEGORICAL:
	print("  == " + cl + " ==")
	model.fit(X, trainingData[cl])
	y_predicted = model.predict(X)
	print(cl + " Accuracy: %.2f" % metrics.accuracy_score(trainingData[cl],y_predicted))
	dump(model, cl +"RawNB.joblib")


print('training complete {:.1f}s'.format(time.time() - startTime))


