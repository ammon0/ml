################################################################################
#
#	Extract LIWC and profile data from the data set, merge the tables, do a
#	stratified split into test and training data, and save them
#
#	Ammon Dodson
#
################################################################################


import sys
import os
import pandas
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

TESTSIZE = 1500

############################### PARSE ARGUMENTS ################################

if len(sys.argv) < 2:
	print("too few arguments")
	exit()

dataPath = sys.argv[1]


############################ BUILD AND VERIFY PATHS ############################

profilePath  = dataPath + '/profile/profile.csv'
liwcPath     = dataPath + '/LIWC/LIWC.csv'

print("profilePath : " + str(profilePath))
print("liwcPath    : " + str(liwcPath))

if( not os.path.isfile(profilePath)):
	print("profile.csv does not exists")
	exit()

if( not os.path.isfile(liwcPath)):
	print("LIWC.csv does not exists")
	exit()

############################ LOAD FILES INTO MEMORY ############################

print("reading LIWC table")
liwcTable    = pandas.read_csv(liwcPath)
#print(liwcTable)
print("reading profile table")
profileTable = pandas.read_csv(profilePath, index_col=0)
#print(profileTable)

############################# JOIN THE DATA FRAMES #############################

print("combining tables")
# use the user ID as the index
profileTable.set_index('userid', inplace=True)
# matchup profile data with liwc data
unifiedTable = liwcTable.join(profileTable, on='userId')

############################# SPLIT THE TEST DATA ##############################

#features = list(liwcTable.columns[1:])
#classes  = list(profileTable.columns[0:])
#print("features: " + str(features))
#print("classes: " + str(classes))

# FIXME: stratify
trainData, testData = train_test_split(unifiedTable, test_size=TESTSIZE)

#print("trainData: " + str(trainData))
#print("testData: " + str(testData))

################################# WRITE TO CSV #################################

print("writing test file")
with open("liwcTest.csv", "w+", newline='') as outFile:
			testData.to_csv(outFile)

print("writing training file")
with open("liwcTrain.csv", "w+", newline='') as outFile:
			trainData.to_csv(outFile)


