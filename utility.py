################################################################################
#
#	A function to combine all text data into a single table
#
#	Ammon Dodson
#
################################################################################


import os
import pandas


PROFILE_SUFFIX  = '/profile/profile.csv'
RELATION_SUFFIX = '/relation/relation.csv'
LIWC_SUFFIX     = '/LIWC/LIWC.csv'
TEXT_SUFFIX     = '/text/'
IMAGE_SUFFIX    = '/image/'

Y             = ['age','gender','ope','con','ext','agr','neu']
Y_NUMERICAL   = ['age',         'ope','con','ext','agr','neu']
Y_CATEGORICAL = ['ageCat', 'gender']
LIWC    = ['Seg', 'WC', 'WPS', 'Sixltr', 'Dic', 'Numerals', 'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion', 'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig', 'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP', 'AllPct']

LIWC_REDUCED = ['achieve', 'affect', 'AllPct', 'anger', 'Apostro', 'auxverb', 'bio', 'certain', 'Colon', 'Comma', 'conj', 'death', 'Dic', 'discrep', 'excl', 'Exclam', 'family', 'friend', 'hear', 'home', 'i', 'insight', 'leisure', 'money', 'motion', 'number', 'Numerals', 'OtherP', 'Parenth', 'past', 'Period', 'posemo', 'ppron', 'preps', 'present', 'pronoun', 'QMark', 'relativ', 'SemiC', 'sexual', 'shehe', 'Sixltr', 'social', 'swear', 'tentat', 'they', 'time', 'verb', 'WC', 'WPS', 'you']

AGEGROUPS = ["xx-24", "25-34", "35-49", "50-xx"]

def verify(path):
	if( not os.path.isfile(path + PROFILE_SUFFIX)):
		print("profile.csv does not exists")
		return False

	if( not os.path.isfile(path + RELATION_SUFFIX)):
		print("relation.csv does not exists")
		return False

	if( not os.path.isfile(path + LIWC_SUFFIX)):
		print("LIWC.csv does not exists")
		return False
	
	return True


## Load all the text features into a single table for convenience
def loadText(path):
	liwcTable     = pandas.read_csv(path + LIWC_SUFFIX)
	liwcTable.rename(columns={"userId": "userid"}, inplace=True)
	liwcTable['text'] = ''
	
	for userid in liwcTable['userid']:
		with open(
			path + TEXT_SUFFIX + userid + '.txt',
			'r',
			errors = "replace"
		) as textFile:
			rawText = textFile.read()
		
		liwcTable.loc[liwcTable.userid == userid, 'text'] = rawText
	
	return liwcTable


def loadRelation(path):
	return pandas.read_csv(path + RELATION_SUFFIX, index_col=0)

def loadProfile(path):
	return pandas.read_csv(path + PROFILE_SUFFIX, index_col=0)


def combineLIWC(profileTable, liwcTable):
	profileTable.set_index('userid', inplace=True)
	unifiedTable = liwcTable.join(profileTable, on='userid')
	return unifiedTable

def age2group(row):
	if  (row['age']<25): return AGEGROUPS[0]
	elif(row['age']<35): return AGEGROUPS[1]
	elif(row['age']<50): return AGEGROUPS[2]
	else               : return AGEGROUPS[3]

def group2age(row):
	if  (row['ageCat'] == AGEGROUPS[0]): return 20
	elif(row['ageCat'] == AGEGROUPS[1]): return 30
	elif(row['ageCat'] == AGEGROUPS[1]): return 40
	else                               : return 60

def ageCategorize(profileTable):
	df = profileTable.copy()
	
	df['ageCat'] = profileTable.apply(
		lambda row: age2group(row), 
		axis=1
	)
	
	return df

def ageCat2age(profileTable):
	df = profileTable.copy()
	
	df['age'] = profileTable.apply(
		lambda row: group2age(row), 
		axis=1
	)
	
	return df

from sklearn.model_selection import KFold
from sklearn                 import metrics
from numpy                   import sqrt

def kFoldCrossRMSE(model, X, y, folds):
	kf = KFold(folds,True)
	mean = 0.0
	
	for train_index, test_index in kf.split(X):
		Xtrain = X.iloc[train_index]
		Xtest  = X.iloc[test_index]
		yTrain = y.iloc[train_index]
		yTest  = y.iloc[test_index]
	
		model.fit(Xtrain, yTrain)
	
		rmse = sqrt(metrics.mean_squared_error(yTest,model.predict(Xtest)))
		#print(accuracy)
		mean += rmse
	
	mean /= folds
	print("mean RMSE: " + str(mean))

def kFoldCrossAccuracy(model, X, y, folds):
	kf = KFold(folds,True)
	mean = 0.0
	
	for train_index, test_index in kf.split(X):
		Xtrain = X.iloc[train_index]
		Xtest  = X.iloc[test_index]
		yTrain = y.iloc[train_index]
		yTest  = y.iloc[test_index]
	
		model.fit(Xtrain, yTrain)
	
		accuracy = metrics.accuracy_score(yTest,model.predict(Xtest))
		#print(accuracy)
		mean += accuracy
	
	mean /= folds
	print("mean accuracy: " + str(mean))


