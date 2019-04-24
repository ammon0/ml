
import pandas
from sklearn import metrics
from joblib import dump, load

TESTING = False


def classify(profileTable,liwcTable, textPath):
	liwcTable.set_index('userid', inplace=True)
	result = pandas.DataFrame(index=liwcTable.index)
	
	
	genderTree = load('text/genderTree.joblib')
	result['gender'] = genderTree.predict(liwcTable)
	
	
	if TESTING:
		copy = profileTable.set_index('userid')
		copy.sort_index(inplace=True)
		
		result.sort_index(inplace=True)
		
		print(metrics.accuracy_score(copy['gender'],result['gender']))
	
	
	return result

