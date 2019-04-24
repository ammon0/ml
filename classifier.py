
import pandas

import baseline

header = ['userid','age','gender','ope','con','ext','agr','neu']

##	The baseline classifier
#	@param profileTable the profile table as a DataFrame
#	@param liwcTable the LIWC table as a DataFrame
#	@param relationTable the relation table as a DataFrame
#	@param imagePath a string with the image directory
#	@param textPath a string with the text directory
#	@returns a profile table as a DataFrame
def base(profileTable,liwcTable,relationTable,imagePath,textPath):
	#results = pandas.DataFrame(profileTable)
	
	results = profileTable.apply(lambda x: 
		[x['userid'],
		baseline.MEDIAN_AGE, 
		baseline.MEDIAN_GENDER,
		baseline.MEAN_OPEN, 
		baseline.MEAN_CON, 
		baseline.MEAN_EXT, 
		baseline.MEAN_AGR, 
		baseline.MEAN_NEU],result_type='broadcast', axis=1)
	
	return results






# change this to use a different classifier
classify = base

