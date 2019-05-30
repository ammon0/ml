################################################################################
#
#	Raw text training
#
################################################################################





################################################################################
#                                 COLLECT DATA
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

trainingData = utility.combineLIWC(pt, utility.loadText(dataPath))
print('data loaded')


################################################################################
#                              TEXT PREPROCESSING
################################################################################


sys.path.insert(0, './text/')
import textPreProcess as tpp

#trainingData = tpp.splitStatusUpdates(trainingData)

print('updates split, lowercasing')

#99655 features
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

print('preprocess complete')


################################################################################
#                               COUNT VECTORIZER
################################################################################


from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

cv = CountVectorizer(ngram_range=(1, 1)).fit(trainingData['status'])
print(len(cv.get_feature_names()))
#print(cv.get_stop_words())

dump(cv, "countVector.joblib")

#X = trainingData[utility.LIWC + ['status']]

print('vectorized')


################################################################################
#                                 TRAINING
################################################################################


import pandas

sparseMatrix = cv.fit_transform(trainingData['status'])
#print(X[0])

#sparseMatrix.shape()

#print(str(sparseMatrix.shape()[0]) + " x " + str(sparseMatrix.shape()[1]))
#print(sparseMatrix.nnz)

#X = pandas.DataFrame(sparseMatrix.toarray(), columns=cv.get_feature_names())

print(sparseMatrix)

print('\nGAUSSIAN NAIVE BAYES\n')
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
print(model.get_params())

model = model.fit(sparseMatrix.toarray(),trainingData['gender'])

accuracy = metrics.accuracy_score(trainingData['gender'],model.predict(sparseMatrix))
print(accuracy)

#for cl in utility.Y_CATEGORICAL:
#	print("  == " + cl + " ==")
#	utility.kFoldCrossAccuracy(model, X, trainingData[cl], 10)

#print('\nMULTINOMIAL NAIVE BAYES\n')
#from sklearn.naive_bayes import MultinomialNB

#model = MultinomialNB()
#print(model.get_params())

#for cl in utility.Y_CATEGORICAL:
#	print("  == " + cl + " ==")
#	utility.kFoldCrossAccuracy(model, X, trainingData[cl], FAST_FOLDS)

#print('\nCOMPLEMENT NAIVE BAYES\n')
#from sklearn.naive_bayes import ComplementNB

#model = ComplementNB(norm=True)
#print(model.get_params())

#for cl in utility.Y_CATEGORICAL:
#	print("  == " + cl + " ==")
#	utility.kFoldCrossAccuracy(model, X, trainingData[cl], FAST_FOLDS)


