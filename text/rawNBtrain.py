################################################################################
#
#	Raw text training
#
################################################################################


from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn                 import metrics
from numpy                   import sqrt
import numpy

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
#pt = utility.ageCategorize(pt)

table = utility.combineLIWC(pt, utility.loadText(dataPath))

if table.isnull().values.any():
	print('combined table has missing values')

print('data loaded {:.1f}s'.format(time.time() - startTime))


################################################################################
#                              TEXT PREPROCESSING
################################################################################


print('preprocessing')

#table['text'].apply(lambda txt: print(">>> " +txt))

table = tpp.splitStatusUpdates(table) #99655 features

table['status'] = table['status'].apply( # 99655 features
	lambda t: t.lower())

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('start ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 98254 features
	lambda t: tpp.removeURL(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('without URLs ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 95171 features
	lambda t: tpp.reduceRepetition(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('without repeats ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 94971 features
	lambda t: tpp.replaceEmojies(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('normalized emoji ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 94822 features
	lambda t: tpp.laughReduction(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('normalized laughs ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 94748 features
	lambda t: tpp.normalizeMoney(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('normalized money ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 94748 features
	lambda t: tpp.normalize(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('normalize dictionary ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 91392 features
	lambda t: tpp.noiseRemoval(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('noise removed ' + str(len(cv.get_feature_names())))

table['status'] = table['status'].apply( # 72581 features
	lambda t: tpp.stemmer(t))

#cv = CountVectorizer(ngram_range=(1, 1))
#cv.fit(table['status'])
#print('porter stemmed ' + str(len(cv.get_feature_names())))

#table['status'].apply(lambda txt: print(">>> " +txt))
#exit()

if table.isnull().values.any():
	print('combined table has missing values after preprocessing')

print('preprocess complete {:.1f}s'.format(time.time() - startTime))


################################################################################
#                               COUNT VECTORIZER
################################################################################


#cv = CountVectorizer(ngram_range=(1, 1),min_df=3)
#cv.fit(table['status'])
#print('df_min=3 ' + str(len(cv.get_feature_names())))

cv = CountVectorizer(ngram_range=(1, 3),min_df=3)
cv.fit(table['status'])
print(len(cv.get_feature_names()))


dump(cv, "countVector.joblib")


print('vectorized {:.1f}s'.format(time.time() - startTime))


################################################################################
#                                 EVALUATION
################################################################################


model = MultinomialNB()
X = cv.transform(table['status'])

print(type(X))
if numpy.isnan(X).any():
	print("NaN's")

kf = KFold(10,True)


mean = 0.0
	
for train_index, test_index in kf.split(X):
	Xtrain = X[train_index]
	Xtest  = X[test_index]
	yTrain = table['gender'].loc[train_index]
	yTest  = table['gender'].loc[test_index]

	model.fit(Xtrain, yTrain)

	accuracy = metrics.accuracy_score(yTest,model.predict(Xtest))
	print(accuracy)
	mean += accuracy

mean /= folds
print("mean accuracy: " + str(mean))

#def kFoldCrossAccuracy(model, X, y, folds):
#	
#	
#	
	

#for cl in utility.Y_CATEGORICAL:
#	print("  == " + cl + " ==")
#	utility.kFoldCrossAccuracy(model, X, table[cl], 10)


################################################################################
#                                 TRAINING
################################################################################


for cl in utility.Y_CATEGORICAL:
	print("  == " + cl + " ==")
	model.fit(X, table[cl])
	y_predicted = model.predict(X)
	print(cl + " Accuracy: %.2f" % metrics.accuracy_score(table[cl],y_predicted))
	dump(model, cl +"RawNB.joblib")


print('training complete {:.1f}s'.format(time.time() - startTime))


