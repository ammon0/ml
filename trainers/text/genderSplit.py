import random
import pandas
import numpy

PROFILEPATH = "/data/training/profile/profile.csv"

TESTSIZE = 1000


pTable = pandas.read_csv(PROFILEPATH, index_col=0)

print(pTable)

# ???
gTable = pTable.loc[:,['userid', 'gender']]
#pTable = pTable.loc['gender']

print(gTable)

length = len(gTable)

print(length)

all_Ids = numpy.arange(length)

print(all_Ids)
