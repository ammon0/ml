
import baseline

##	The baseline classifier
#	@param image the open jpg file handle
#	@param text The open text file handle
#	@param licw A list containing the user's LIWC data as strings
#	#param likes a list containing like ID's as strings
#	@returns 7 floats indicating age, gender (0:male), o, c, e, a, n
def base(image, text, liwc, likes):
	return (
		baseline.MEDIAN_AGE, 
		baseline.MEDIAN_GENDER,
		baseline.MEAN_OPEN, 
		baseline.MEAN_CON, 
		baseline.MEAN_EXT, 
		baseline.MEAN_AGR, 
		baseline.MEAN_NEU
	)






# change this to use a different classifier
classify = base

