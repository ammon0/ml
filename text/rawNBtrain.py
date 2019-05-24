################################################################################
#
#	Raw text training
#
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
print(pt)

trainingData = utility.combineLIWC(pt, utility.loadText(dataPath))

