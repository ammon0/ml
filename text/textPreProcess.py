################################################################################
#
#	Text preprocessing routines
#
#	By Ammon Dodson
#
################################################################################

import re
import pandas
#from spellchecker import SpellChecker

# from: https://gist.github.com/sebleier/554280
STOP_WORDS = [
	'i', "i'm", 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
	"you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
	'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
	'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
	'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
	'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
	'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
	'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
	'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
	'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
	'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
	'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
	'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
	'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
	"don't", 'should', "should've", 'now', "aren't", "couldn't", "didn't",
	"doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
	"needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]

# partially taken from https://en.wikipedia.org/wiki/List_of_emoticons
POS_EMOTE = [
	':)',':d',';)',';p','<3','^.^',':)','=d',':-)',':-]',':]', ':-3',':3',
	':->',':>','8-)','8)',':-}',':}',':o)',':c)',':^)','=]','=)',':d','8d','xd',
	'xd','=d','=3',':-))','d:<','d:','d8','d;','d=','dx',':p',':p',
	':þ',':þ',':b','d:','=p','>:p',':-*',':*',':×',';)','*-)','*)', ';]',
	';^)',';d','o:)','0:3','0:)','0;^)','>:)','}:)','3:)','>;)',":')",";-)",
	"&hearts;","d':",'(:','[:','[x','^^'
]

NEG_EMOTE = [
	'-__-','-_-',":'(",':[',':-||','>:[',':{',':(',':c',':<','>:(',':l',
	'=l',':s',':|',':$',':x',':o',':o',':-0','>:o',':/','>:\\','>:/',
	':\\','=/','=\\',':@','=[','grr','meh','=(','=x','-.-',
	'/:',']:','<_<','>_>','t.t','t-t','</3','>_<','>d'
]

OTHER_EMOTE =['o_o','o.0','0.o','o.o','woo','yay']

WORD_SEP = r"[\s,.]+"

NORM_DIC = {
	# wont work with '
	'b-day':'birthday','b-days':'birthdays','bday':'birthday',
	'cuz':'because','bc':'because','coz':'because',
	'u':'you','uu':'you','ya':'you',
	'sum1':'someone','som1':'someone',
	'tmr':'tomorrow','tmrw':'tomorrow','2mrw':'tomorrow',
	'thanx':'thanks','tnx':'thanks',
	'plz':'please','pls':'please',
	'&':'and','n':'and','r':'are','w':'with','w/':'with',
	'y':'why','nah':'no',
	'fb':'facebook','acc':'account',"'net":'internet','wats':'whats',
	'yr':'year', 'hw':'homework',
	'bout':'about','dam':'damn',
	'uni':'university',
	'lolz':'lol','lulz':'lol','yayz':'yay',
	'b4':'before','tho':'though','+':'and','=':'is','postn':'posting',
	'hv':'have','ppl':'people','bby':'baby',
	'f-ing':'fucking','a$$':'ass',
	'x-mas':'christmas',
	'1':'one','4':'for'
}
normalizeRE = re.compile(
	'\s+('+('|'.join(map(re.escape, NORM_DIC)))+')'+WORD_SEP
)

#"it's":'it is',"there's":'there is',
#	"doesn't":'does not',
#	"won't":'will not','cannot':'can not',"wasn't":'was not',"i've":'i have',
#	"i'll":'i will',"i'v":'i have',"don't":'do not',"they're":'they are',
#	"you're":'you are','theyre':'they are'

#print(normalizeRE)
#print(normalizeRE.groups)

wordSplitRE = re.compile(WORD_SEP)

SU_DIVIDER = "<><><>"

WORD_LIST = [
	'posemote','negemote','icky','m&ms','','facebook','psy','forza',
	'canceled','uploading','hmph','fml','wtf','youtube','gonna'
]

#spell = SpellChecker()
#spell.word_frequency.load_words(WORD_LIST)


def splitStatusUpdates(table):
	# remove repeated commas
	txt = table['text'].apply(lambda t: re.sub(',{2,}', ',', t))
	# a new column containing a list of status updates
	updateList = txt.apply(lambda t: re.split(',(?=[^ ])', t))
	
	
	# from: https://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows/17116976#17116976
	updates = updateList.apply(pandas.Series, 1).stack()
	updates.index = updates.index.droplevel(-1)
	updates.name = 'status'
	
	table = table.join(updates)
	del table['text']
	return table

def removeURL(txt):
	return re.sub('(http|www)[^ ]+',' ', txt)

def laughReduction(txt):
	txt = re.sub('(mw)?a?(ha){2,}','lol',txt)
	txt = re.sub('(he){2,}','lol',txt)
	#txt = re.sub('ha+|wo+h?','haha',txt)
	return txt

## Remove character repetition greater than 2 times
#	should happend before emote replacement
def reduceRepetition(txt):
	return re.sub(r'(.)\1{2,}',r'\1\1',txt)

def replaceEmojies(txt):
	for emoji in POS_EMOTE:
		txt = txt.replace(emoji, ' posemote ')
	for emoji in NEG_EMOTE:
		txt = txt.replace(emoji, ' negemote ')
	for emoji in OTHER_EMOTE:
		txt = txt.replace(emoji, ' exclaim ')
	
	txt = txt.replace('@',' at ')
	
	txt = txt.replace('!!', ' exclaim exclaim ')
	txt = txt.replace('!', ' exclaim ')
	
	txt = txt.replace('??', ' question question ')
	txt = txt.replace('?', ' question ')
	
	return txt

# from: https://www.daniweb.com/programming/software-development/code/216636/multiple-word-replace-in-text-python
def normalize(txt):
	"""
	take a text and replace words that match a key in a dictionary with
	the associated value, return the changed text
	"""
	def translate(match):
		return ' ' + NORM_DIC[match.group(1)] + ' '
	
	newTxt = normalizeRE.sub(translate, txt)
	newTxt = normalizeRE.sub(translate, newTxt)
	
#	if(newTxt != txt):
#		print(txt)
#		print("\n"+newTxt+"\n\n")
	
	return newTxt

def normalizeMoney(txt):
	txt = re.sub('\$[0-9]+',' $$ ',txt)
	txt = re.sub('[0-9]+k',' $$ ',txt)
	return txt


def noiseRemoval(txt):
	txt = re.sub(r"[^a-z $']",' ',txt)
	
	# single quotes
	txt = re.sub("'",'',txt)
	
	return txt


def spellCorrect(txt):
	def correct(word):
		if spell.unknown(word): return spell.correction(word)
		else: return word
	
	l = txt.split(' ')
	
	#print("old: " +str(l))
	
	l = [correct(word) for word in l if word != '']
	
	#print("new: " + str(l))
	
	return ' '.join(l)

import porterStemmer

def stemmer(line):
	output = ''
	word = ''
	p = porterStemmer.PorterStemmer()
	for c in line:
		if c.isalpha():
			word += c.lower()
		else:
			if word:
				output += p.stem(word, 0,len(word)-1)
				word = ''
			output += c.lower()
	return output



def removeStopWords(txt):
	for w in STOP_WORDS:
		txt = txt.replace(' '+w+' ', ' ')
	return txt









