
import pandas
import baseline
import utility
from sklearn import metrics
from joblib import load
from statistics import mode
from numpy import sqrt

TESTING = False


def genderTree(profileTable,liwcTable, modulePath):
	#liwcTable.set_index('userid', inplace=True)
	#result = pandas.DataFrame(index=liwcTable.index)
	result = pandas.DataFrame(
		index=liwcTable['userid'],
		columns=utility.Y_NUMERICAL
	)
	
	# try logistic regression
	
	genderTree = load(modulePath + '/text/genderTree.joblib')
	result['gender'] = genderTree.predict(liwcTable[utility.LIWC])
	
	
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
	results['age']    = baseline.MEDIAN_AGE
	results['ope']    = baseline.MEAN_OPEN
	results['con']    = baseline.MEAN_CON
	results['ext']    = baseline.MEAN_EXT
	results['agr']    = baseline.MEAN_AGR
	results['neu']    = baseline.MEAN_NEU
	
	model = load(modulePath + '/text/linearRegression.joblib')
	results[utility.Y_NUMERICAL] = model.predict(liwcTable[utility.LIWC])
	
	if TESTING:
		copy = profileTable.set_index('userid')
		copy.sort_index(inplace=True)
		
		results.sort_index(inplace=True)
		
		for c in utility.Y_NUMERICAL:
			print(c + " RMSE: " + str(sqrt(metrics.mean_squared_error(
				copy[c],
				results[c]
			))))
	
	return results


def liwcLogReg(profileTable, liwcTable, modulePath):
	results = pandas.DataFrame(
		index=liwcTable['userid'],
		columns=utility.Y_CATEGORICAL
	)
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	
	genderModel = load(modulePath + '/text/genderLogReg.joblib')
	ageModel    = load(modulePath + '/text/ageLogReg.joblib')
	
	results['ageCat'] = ageModel.predict(liwcTable[utility.LIWC])
	results['gender'] = genderModel.predict(liwcTable[utility.LIWC])
	
	#print(results)
	
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
	
	results = utility.ageCat2age(results)
	
	#print(results)
	
	return results

def rawText(profileTable, textTable, modulePath):
	from sys import path
	path.insert(0,modulePath + '/text/')
	import textPreProcess as tpp
	
	results = pandas.DataFrame(
		index=profileTable['userid'],
		columns=utility.Y_CATEGORICAL
	)
	
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	
	textTable = tpp.splitStatusUpdates(textTable)
	
	textTable['status'] = textTable['status'].apply(
		lambda t: t.lower())
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.removeURL(t))
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.reduceRepetition(t))
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.replaceEmojies(t))
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.laughReduction(t))
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.normalizeMoney(t))
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.normalize(t))
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.noiseRemoval(t))
	textTable['status'] = textTable['status'].apply(
		lambda t: tpp.stemmer(t))
	
	#textTable['status'].apply(lambda txt: print(">>> " +txt))
	
	cv          = load(modulePath + '/text/countVector.joblib')
	genderModel = load(modulePath + '/text/genderRawNB.joblib')
	ageModel    = load(modulePath + '/text/ageRawNB.joblib')
	
	
	X = cv.transform(textTable['status'])
	
	textTable['gender'] = genderModel.predict(X)
	textTable['age']    = ageModel   .predict(X)
	
	# ensamble status results for each user
	for i in range(0,textTable.index[-1]):
		uId = textTable.loc[i]['userid']
		try: uId = uId.iloc[0]
		except: pass
		
		try   : g = mode(textTable.loc[i]['gender'])
		except: pass # leave the baseline in case of tie
		
		# mode is probably not adequate for age categories
		try   : a = mode(textTable.loc[i]['age'].astype('int'))
		except: a = 30 # in case of tie, favor second most common age group??
		
		results.loc[uId,'gender'] = g
		results.loc[uId,'age'] = a
	
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
		
#		print("gender accuracy: " + str(metrics.accuracy_score(
#			copy['gender'].astype('int'),
#			results['gender'].astype('int')
#		)))
#		
#		print("age MSE: " + str(metrics.mean_squared_error(
#			copy['age'],
#			results['age']
#		)))
	
	return(results)







