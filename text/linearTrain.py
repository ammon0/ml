################################################################################
#
#	Train a decision tree from data
#
#	Ammon Dodson
#
################################################################################


import pandas
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from joblib import dump, load

classes = ['age','ope','con','ext','agr','neu']

trainingData = pandas.read_csv("liwcTrain.csv", index_col=0)
trainingData.drop('userId', axis=1, inplace=True)

features = list(trainingData.columns[:82])
X = trainingData[features]
y = trainingData[classes]

#################################### TRAIN #####################################

model = LinearRegression().fit(X, y)

##################################### TEST #####################################

testData = pandas.read_csv('liwcTest.csv', index_col=0)

X = testData[features]
y = testData[classes]

p = model.predict(X)

p = pandas.DataFrame(data=p, columns=classes)

print(y['age'])
print(p['age'])

for i in classes:
	print(str(i) + ': ' + str(metrics.mean_squared_error(y[i],p[i])))


#print(metrics.mean_squared_error(y,p))

################################## SAVE MODEL ##################################

dump(model, "linearRegression.joblib")
