

##	The baseline classifier
#	@param image the open jpg file handle
#	@param text The open text file handle
#	@param licw A list containing the user's LIWC data as strings
#	#param likes a list containing like ID's as strings
#	@returns 7 floats indicating age, gender (0:male), o, c, e, a, n
def baseline(image, text, liwc, likes):
	return (0, 0, 3.908690526315789, 3.4456168421052626, 3.486857894736842, 3.5839042105263155, 2.732424210526316)






# change this to use a different classifier
classify = baseline

