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

CLASSES = ['age','gender','ope','con','ext','agr','neu']
LIWC    = ['Seg', 'WC', 'WPS', 'Sixltr', 'Dic', 'Numerals', 'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion', 'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig', 'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP', 'AllPct']


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


def loadLIWC(path):
	liwcTable     = pandas.read_csv(path + LIWC_SUFFIX)
	liwcTable.rename(columns={"userId": "userid"}, inplace=True)
	
	return liwcTable

def loadRelation(path):
	return pandas.read_csv(path + RELATION_SUFFIX, index_col=0)

def loadProfile(path):
	return pandas.read_csv(path + PROFILE_SUFFIX, index_col=0)


def combineLIWC(profileTable, liwcTable):
	profileTable.set_index('userid', inplace=True)
	unifiedTable = liwcTable.join(profileTable, on='userid')
	return unifiedTable


