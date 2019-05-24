
import pandas
import utility
from sklearn import metrics
from joblib import load

TESTING = False


def genderTree(profileTable,liwcTable):
	liwcTable.set_index('userid', inplace=True)
	result = pandas.DataFrame(index=liwcTable.index)
	
	# try logistic regression
	
	genderTree = load('/home/itadmin/ml/text/genderTree.joblib')
	result['gender'] = genderTree.predict(liwcTable)
	
	
	if TESTING:
		copy = profileTable.set_index('userid')
		copy.sort_index(inplace=True)
		
		result.sort_index(inplace=True)
		
		print(metrics.accuracy_score(copy['gender'],result['gender']))
	
	
	return result


def liwcLinReg(profileTable, liwcTable, modulePath):
	results = pandas.DataFrame(
		index=liwcTable['userid'],
		columns=utility.Y_NUMERICAL
	)
	
	model = load(modulePath + '/text/linearRegression.joblib')
	results[utility.Y_NUMERICAL] = model.predict(liwcTable[utility.LIWC])
	
	if TESTING:
		copy = profileTable.set_index('userid')
		copy.sort_index(inplace=True)
		
		results.sort_index(inplace=True)
		
		for c in utility.Y_NUMERICAL:
			print(c + " RMSE: " + str(metrics.mean_squared_error(copy[c],results[c])))
	
	return results


def liwcLogReg(profileTable, liwcTable, modulePath):
	results = pandas.DataFrame(
		index=liwcTable['userid'],
		columns=utility.Y_CATEGORICAL
	)
	
	genderModel = load(modulePath + '/text/genderLogReg.joblib')
	ageModel    = load(modulePath + '/text/ageLogReg.joblib')
	
	results['ageCat'] = ageModel.predict(liwcTable[utility.LIWC])
	results['gender'] = genderModel.predict(liwcTable[utility.LIWC])
	
	print(results)
	
	if TESTING:
		copy = profileTable.set_index('userid')
		copy.sort_index(inplace=True)
		
		copy = utility.ageCategorize(copy)
		
		results.sort_index(inplace=True)
		
		for c in utility.Y_CATEGORICAL:
			print(c + " accuracy: " + str(metrics.accuracy_score(
				copy[c],
				results[c]
			)))
	
	
	return utility.ageCat2age(results)

def rawText(profileTable, textTable, modulePath):
	from sys import path
	path.insert(0,modulePath + '/text/')
	import textPreProcess as tpp
	
	#print(textTable)
	
	print(tpp.normalize(" there's "))
	print(tpp.normalize(" i'mma "))
	
	textTable['text'] = textTable['text'].apply(lambda t: tpp.divideUpdates(t))
	
	# lowercase
	textTable['text'] = textTable['text'].apply(lambda t: t.lower())
	
	textTable['text'] = textTable['text'].apply(lambda t: tpp.removeURL(t))
	
	
	textTable['text'] = textTable['text'].apply(
		lambda t: tpp.reduceRepetition(t)
	)
	textTable['text'] = textTable['text'].apply(lambda t: tpp.replaceEmojies(t))
	
	textTable['text'] = textTable['text'].apply(lambda t: tpp.laughReduction(t))
	
	textTable['text'] = textTable['text'].apply(lambda t: tpp.normalizeMoney(t))
	
	textTable['text'] = textTable['text'].apply(lambda t: tpp.normalize(t))
	
	textTable['text'] = textTable['text'].apply(
		lambda t: tpp.noiseRemoval(t)
	)
#	
#	textTable['text'].apply(
#		lambda t: tpp.spellCorrect(t)
#	)
	
#	textTable['text'] = textTable['text'].apply(
#		lambda t: tpp.removeStopWords(t)
#	)
#	textTable['tList'] = textTable['text'].apply(
#		lambda t: tpp.splitStatusUpdates(t)
#	)
	
	textTable['text'].apply(lambda txt: print(txt + '\n'))
	
	#print(textTable['tList'])







