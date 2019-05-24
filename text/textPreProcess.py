################################################################################
#
#	Text preprocessing routines
#
#	By Ammon Dodson
#
################################################################################

import re
from spellchecker import SpellChecker

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
	'xd','=d','=3',':-))','d:<','d:','d8','d;','d=','dx',':p','xp','xp',':p',
	':þ',':þ',':b','d:','=p','>:p',':-*',':*',':×',';)','*-)','*)', ';]',
	';^)',';d','o:)','0:3','0:)','0;^)','>:)','}:)','3:)','>;)',":')",";-)",
	"&hearts;",'t.t',"d':",'(:','[:','[x','^^'
]

NEG_EMOTE = [
	'-__-',":'(",':[',':-||','>:[',':{',':(',':c',':<','>:(',':l',
	'=l',':s',':|',':$',':x',':o',':o',':-0','>:o',':/','>:\\','>:/',
	':\\','=/','=\\',':@','=[','grr','meh','=(','=x','-.-',
	'/:',']:',
]

OTHER_EMOTE =['o_o','o.0','0.o','o.o','<_<','>_>','woo','yay']

WORD_SEP = r"[\s,.]+"

NORM_DIC = {
	'1':'one','y':'why','nah':'no','sum1':'someone',
	'fb':'facebook','acc':'account',"'net":'internet','wats':'what is',
	'u':'you','uu':'you','ya':'you',
	'yr':'year', '&':'and','hw':'homework',
	'thanx':'thanks','tnx':'thanks',
	'ima':'i will',"i'mma":'i will','gona':'will','gonna':'will',
	'cwap':'crap','bout':'about','dam':'damn',
	'uni':'university',
	'b-day':'birthday','b-days':'birthdays',
	'lolz':'lol','lulz':'lol','yayz':'yay',
	'tmr':'tomorrow','tmrw':'tomorrow',
	'cuz':'because','bc':'because',
	'b4':'before','tho':'though','+':'and','=':'is','postn':'posting',
	"pj's":'pajamas','cant':"can't",'hv':'have','ppl':'people','bby':'baby',
	"playin'":'playing',
	'f-ing':'fucking','a$$':'ass',
	'x-mas':'christmas',
	"it's":'it is',"there's":'there is',"i'm":'i am',"doesn't":'does not',
	"won't":'will not','cannot':'can not',"wasn't":'was not',"i've":'i have',
	"i'll":'i will',"i'v":'i have',
	"you're":'you are'
}
normalizeRE = re.compile(
	WORD_SEP+'('+('|'.join(map(re.escape, NORM_DIC)))+')'+WORD_SEP
)

#print(normalizeRE)
#print(normalizeRE.groups)

wordSplitRE = re.compile(WORD_SEP)

SU_DIVIDER = "\n<><><> "

WORD_LIST = [
	'<><><>','posemote','negemote','icky','m&ms','','facebook','psy','forza',
	'canceled','uploading','hmph','fml','wtf','youtube'
]

spell = SpellChecker()
spell.word_frequency.load_words(WORD_LIST)


def divideUpdates(txt):
	txt = SU_DIVIDER + txt
	txt = re.sub(',{2,}', ',', txt)
	return re.sub(',(?=[^ ])', SU_DIVIDER, txt)

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
	
#	if(newTxt != txt):
#		#print("\n\n" + txt + "\n")
#		print("\n"+newTxt+"\n\n")
	
	return newTxt

def normalizeMoney(txt):
	txt = txt = re.sub('\$[0-9]+',' $$ ',txt)
	txt = txt = re.sub('[0-9]+k',' $$ ',txt)
	return txt

##	Remove the escape character from newlines in the status updates
#	Should follow emote conversion
#def removeEscapes(txt):
#	txt = txt.replace('\\\n','\n')
#	return txt.replace('\\',' ')

def noiseRemoval(txt):
	txt = re.sub(r'([]0-9@%^*+=,./\\:;"~`_|(){}[-])+',' ',txt)
	
	# single quotes
	txt = re.sub("' ",' ',txt)
	txt = re.sub(" '",' ',txt)
	
	return txt


def spellCorrect(txt):
	l = wordSplitRE.split(txt)
	
	misspelled = spell.unknown(l)
	
	if len(misspelled) > 0:
		print(txt)
	
	for word in misspelled:
		# Get the one `most likely` answer
		print('"'+ word + '"' + " -> " + spell.correction(word))



def removeStopWords(txt):
	for w in STOP_WORDS:
		txt = txt.replace(' '+w+' ', ' ')
	return txt


##	Split text into individual status updates.
#
#	The status updates appear to be separated by a comma without a space, whilst a
#	user will typically place a space after commas
def splitStatusUpdates(txt):
	l = re.split(SU_DIVIDER, txt)
	#del l[-1]
	return l






