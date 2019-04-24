

import pandas
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


dataTable = pandas.read_csv("liwcTrain.csv", index_col=0)

dataTable.drop('userId', axis=1, inplace=True)

print(dataTable)

############################ LOAD FILES INTO MEMORY ############################

liwcTable    = pd.read_csv(liwcPath)
profileTable = pd.read_csv(profilePath, index_col=0)

#print(liwcTable)
#print(profileTable)

############################# JOIN THE DATA FRAMES #############################

# use the user ID as the index
profileTable.set_index('userid', inplace=True)
# matchup profile data with liwc data
unifiedTable = liwcTable.join(profileTable, on='userId')

#print(unifiedTable)

features = list(liwcTable.columns[1:])

#################################### GENDER ####################################

X      = unifiedTable[features]
gender = unifiedTable["gender"]

for i in range(1,40):


	# Train a decision tree and compute its training accuracy
	genderTree = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy')
	# ineffective: 2

	mean = 0.0

	# 10-fold
	skf = StratifiedKFold(n_splits=10)
	skf.get_n_splits(X,gender)
	for train_index, test_index in skf.split(X,gender):
		Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
		yTrain, yTest = gender.iloc[train_index], gender.iloc[test_index]
	
		genderTree.fit(X, gender)
	
		accuracy = metrics.accuracy_score(yTest,genderTree.predict(Xtest))
		#print(accuracy)
		mean += accuracy

	print(str(i) + ": mean: " + str(mean/10))

genderTree.fit(X, gender)
print(metrics.accuracy_score(gender,genderTree.predict(X)))

