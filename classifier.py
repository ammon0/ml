
import pandas

import baseline
import textClassifier
#import age_likesClassifier
#import gender_likesClassifier
#import imageClassifier
#header = ['userid','age','gender','ope','con','ext','agr','neu']

##	The baseline classifier
#	@param profileTable the profile table as a DataFrame
#	@param liwcTable the LIWC table as a DataFrame
#	@param relationTable the relation table as a DataFrame
#	@param imagePath a string with the image directory
#	@param textPath a string with the text directory
#	@returns a profile table as a DataFrame
def base(profileTable,textTable,relationTable,imagePath,modulePath):
	results = pandas.DataFrame(index=profileTable['userid'])
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	results['ope']    = baseline.MEAN_OPEN
	results['con']    = baseline.MEAN_CON
	results['ext']    = baseline.MEAN_EXT
	results['agr']    = baseline.MEAN_AGR
	results['neu']    = baseline.MEAN_NEU
	
	return results


##	The baseline classifier
#
#	Results of classifiers should be indexed by userid so that they can be
#	combined
#
#	@param profileTable the profile table as a DataFrame
#	@param liwcTable the LIWC table as a DataFrame
#	@param relationTable the relation table as a DataFrame
#	@param imagePath a string with the image directory
#	@param textPath a string with the text directory
#	@returns a profile table as a DataFrame
def week4(profileTable,textTable,relationTable,imagePath,modulePath):
	results = pandas.DataFrame(index=profileTable['userid'])
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	results['ope']    = baseline.MEAN_OPEN
	results['con']    = baseline.MEAN_CON
	results['ext']    = baseline.MEAN_EXT
	results['agr']    = baseline.MEAN_AGR
	results['neu']    = baseline.MEAN_NEU
	
	
	textResults = textClassifier.genderTree(profileTable,liwcTable)
	results = results.assign(gender=textResults['gender'])
	
	return results



def jakeTesting(profileTable,textTable,relationTable,imagePath,modulePath):
	#results = profileTable
	#results = profileTable
	#print('profileTable')
	#print(profileTable)
	results = pandas.DataFrame(index=profileTable['userid'])
	
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	results['ope']    = baseline.MEAN_OPEN
	results['con']    = baseline.MEAN_CON
	results['ext']    = baseline.MEAN_EXT
	results['agr']    = baseline.MEAN_AGR
	results['neu']    = baseline.MEAN_NEU
	#print('0')
	#print(results)
	likeResults = gender_likesClassifier.likeLogReg(profileTable,relationTable)
	#print('likeResults')
	#print(likeResults)
	results['gender'] = likeResults['gender']
	#print('5')
	#print(results)
	#print(results.count())
	return results


def mattTesting(profileTable,textTable,relationTable,imagePath,modulePath):
        #results = profileTable
        #results = profileTable
        #print('profileTable')
        #print(profileTable)
        results = pandas.DataFrame(index=profileTable['userid'])
        
        results['age']    = baseline.MEDIAN_AGE
        results['gender'] = baseline.MEDIAN_GENDER
        results['ope']    = baseline.MEAN_OPEN
        results['con']    = baseline.MEAN_CON
        results['ext']    = baseline.MEAN_EXT
        results['agr']    = baseline.MEAN_AGR
        results['neu']    = baseline.MEAN_NEU
        #print('0')
        #print(results)
        #likeResults = gender_likesClassifier.likeLogReg(profileTable,relationTable)
        #imageResults = image_classifier.genderNN(profileTable, (imagePath+'/'+'')
	#print('likeResults')
        #print(likeResults)
        #results['gender'] = imageResults['gender']
        #print('5')
        #print(results)
        #print(results.count())

        imageClassifier.imageGender(profileTable,modulePath,imagePath)

        return results



##	The baseline classifier
#
#	Results of classifiers should be indexed by userid so that they can be
#	combined
#
#	@param profileTable the profile table as a DataFrame
#	@param liwcTable the LIWC table as a DataFrame
#	@param relationTable the relation table as a DataFrame
#	@param imagePath a string with the image directory
#	@param textPath a string with the text directory
#	@returns a profile table as a DataFrame
def week5(profileTable,textTable,relationTable,imagePath,modulePath):
	results = pandas.DataFrame(index=profileTable['userid'])
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	results['ope']    = baseline.MEAN_OPEN
	results['con']    = baseline.MEAN_CON
	results['ext']    = baseline.MEAN_EXT
	results['agr']    = baseline.MEAN_AGR
	results['neu']    = baseline.MEAN_NEU
	
	textResults = textClassifier.liwcLinReg(profileTable,liwcTable,modulePath)
	results['age'] = textResults['age']
	results['ope'] = textResults['ope']
	results['con'] = textResults['con']
	results['agr'] = textResults['agr']
	results['ext'] = textResults['ext']
	results['neu'] = textResults['neu']
	
	likeResults = gender_likesClassifier.likeLogReg(profileTable,relationTable)
	results['gender'] = likeResults['gender']
	
	return results


##	The baseline classifier
#
#	Results of classifiers should be indexed by userid so that they can be
#	combined
#
#	@param profileTable the profile table as a DataFrame
#	@param liwcTable the LIWC table as a DataFrame
#	@param relationTable the relation table as a DataFrame
#	@param imagePath a string with the image directory
#	@param textPath a string with the text directory
#	@returns a profile table as a DataFrame
def week6(profileTable,textTable,relationTable,imagePath,modulePath):
	results = pandas.DataFrame(index=profileTable['userid'])
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	results['ope']    = baseline.MEAN_OPEN
	results['con']    = baseline.MEAN_CON
	results['ext']    = baseline.MEAN_EXT
	results['agr']    = baseline.MEAN_AGR
	results['neu']    = baseline.MEAN_NEU
	
	age_likeResults = age_likesClassifier.likeLogReg(profileTable,relationTable)
	results['age'] = age_likeResults['age']

	gender_likeResults = gender_likesClassifier.likeLogReg(profileTable,relationTable)
	results['gender'] = gender_likeResults['gender']
	
	return results


def week9(profileTable,textTable,relationTable,imagePath,modulePath):
	results = pandas.DataFrame(index=profileTable['userid'])
	results['age']    = baseline.MEDIAN_AGE
	results['gender'] = baseline.MEDIAN_GENDER
	results['ope']    = baseline.MEAN_OPEN
	results['con']    = baseline.MEAN_CON
	results['ext']    = baseline.MEAN_EXT
	results['agr']    = baseline.MEAN_AGR
	results['neu']    = baseline.MEAN_NEU
	
	textResults = textClassifier.rawText(profileTable,textTable,modulePath)
	
	return results

# change this to use a different classifier
classify = week9


