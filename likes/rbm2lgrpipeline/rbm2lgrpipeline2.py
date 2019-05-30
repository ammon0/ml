import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy
import pandas
from sklearn.ensemble import RandomForestRegressor
from scipy import sparse
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
import time
# relation.csv
userid  = 'userid'
like_id = 'like_id'

relation_col = [userid, like_id]

# profile.csv
age     = 'age'
gender  = 'gender'
ope     = 'ope'
con     = 'con'
ext     = 'ext'
agr     = 'agr'
neu     = 'neu'

profile_col = [userid, age, gender, ope, con, ext, agr, neu]


#ul = pandas.read_csv('relation.csv', names = relation_col, low_memory=False, index_col = 0, skiprows = 1)
#pr = pandas.read_csv('profile.csv', names = profile_col,low_memory=False, index_col=0, skiprows = 1)
#li = pandas.read_csv('LIWC.csv', names = LIWC_col, low_memory=False, skiprows = 1)

#print(ul)
#print(pr.shape)
from sklearn.model_selection import train_test_split

#grouped_df = ul.groupby(userid)
#for key, item in grouped_df:
#    print(grouped_df.get_group(key), "\n\n")

def load_data():
    df_relation = pandas.read_csv('relation.csv', names = relation_col, low_memory=False, index_col = 0, skiprows = 1)
    df_profile = pandas.read_csv('profile.csv', names = profile_col,low_memory=False, index_col=0, skiprows = 1)
    #df_LIWC = pandas.read_csv('LIWC.csv', names = LIWC_col, low_memory=False, skiprows = 1)
    # generate summary statistics for trimming
    # might need a better way to do this in the future? (computationally taxing)
    likes_count = df_relation.groupby(userid).count()
    sd = likes_count.std()[0]
    mean = likes_count.mean()[0]
    result = mean + 3 * sd
    # generate hashmap(?)
    likes_keys = df_relation[userid].value_counts().keys().tolist()
    likes_counts = df_relation[userid].value_counts().tolist()
    # drops the rows in the relation dataframe that are 
    # 2 standard deviations away from the mean.
    # These are values we shouldn't expect to see so let's take 
    # them out
    df_merged = pandas.merge(df_profile, df_relation, on=userid,how='left')
    print(df_profile)
    k = 0
    for (key, count) in zip(likes_keys,likes_counts):
        
        if count >= result:
            #if df_merged.loc[df_merged[userid] == key,gender].iloc[0] == 1.0:
            #    k += 1
            idxs_relation = df_relation[df_relation[userid] == key].index
            df_relation.drop(idxs_relation, inplace = True)
            
            idxs_profile = df_profile[df_profile[userid] == key].index
            df_profile.drop(idxs_profile, inplace = True)
            #idxs_LIWC = df_LIWC[df_LIWC[userid] == key].index
            #df_LIWC.drop(idxs_LIWC, inplace = True)
    #print('women' + str(count))
    #print('number of women in outliers ' + str(k))
    #print(df_profile)
    return (df_relation, df_profile)


#(ul, pr) = load_data()
#ul.to_csv('ul.csv')
#pr.to_csv('pr.csv')

#ul =  pandas.read_csv('ul.csv', names = relation_col, low_memory=False, index_col = 0, skiprows = 1)
#pr = pandas.read_csv('pr.csv', names = profile_col,low_memory=False, index_col=0, skiprows = 1)
ul =  pandas.read_csv('relation.csv', names = relation_col, low_memory=False, index_col = 0, skiprows = 1)
pr = pandas.read_csv('profile.csv', names = profile_col,low_memory=False, index_col=0, skiprows = 1)
print('##################################################')
print('loaded models')
print('##################################################')
#print(ul[like_id].max())
#print(ul[like_id].min())
pr = pr[[userid, gender]]
ul[like_id] = ul[like_id].astype(str)
#print(ul)
#print(pr)
#ul = ul.groupby(userid)[like_id].apply(' '.join).reset_index()
ul = ul.groupby('userid',as_index=False).agg({'like_id': lambda x: "%s" % ' '.join(x)})
#print(ul)
#ul = ul.groupby(userid)[like_id].apply(' '.join)

df = pandas.merge(pr, ul, on=userid,how='left')
print('##################################################')
print('merged models')
print('##################################################')
#7443
#1860
#random.shuffle(df)

#df_like = cv.fit_transform(df[like_id])
#numpy.random.shuffle(df.values)

#n = 7443
#test = df[0:n]
#train = df[n:]
#cv = TfidfVectorizer(ngram_range = (1,19),max_features=2**10)
#536204
cv = TfidfVectorizer(max_features=536204)
print('##################################################')
print('count vectorized')
print('##################################################')
X = cv.fit_transform(df[like_id])
y = df[gender]

y_train, y_test, X_train, X_test = train_test_split(y, X, 
													test_size=.2, 
													random_state=1234)
#d = 2**5
#n_estimators=2500, criterion='entropy',max_features='log2', oob_score=True,max_depth=d,min_samples_split=int(numpy.ceil(numpy.sqrt(d))),random_state=150, n_jobs=8
#clf = RandomForestClassifier(n_estimators=2**10, 
#							criterion='entropy',
#							max_features='sqrt', 
#							oob_score=True,
#							max_depth=d,
#							min_samples_split=int(numpy.ceil(numpy.sqrt(d))), 
#							n_jobs=8)


#alpha=0.05
#clf = MultinomialNB(alpha=0.1,
#					class_prior=None,
#					fit_prior=True)

from sklearn.linear_model import LogisticRegression, SGDClassifier
#avg: 0.8301671394468135
#max: 0.8715846994535519
#dual=False, C=1.0, fit_intercept=True, intercept_scaling=1, random_state=1234, max_iter=100, n_jobs=8

#avg: 0.8305753011948461
#max: 0.8729508196721312
#dual=False, C=0.5, fit_intercept=True, intercept_scaling=1, random_state=1234, max_iter=1000, n_jobs=8

#avg: 0.8330343196196113
#max: 0.8743169398907104
#dual=False, C=0.1, fit_intercept=True, intercept_scaling=1, random_state=1234, max_iter=1000, n_jobs=8

#avg: 0.8364453340756599
#max: 0.8770491803278688
#C=.1, fit_intercept=True, intercept_scaling=1, random_state=1234, max_iter=10000, n_jobs=8, solver='saga'

#best model (so far)
#C=.1,						random_state=1234, 						max_iter=10000, 						n_jobs=8,						solver='saga',						penalty='l2'
#lgr = LogisticRegression(C=0.1,
#						random_state=1234, 
#						max_iter=10000, 
#						n_jobs=8,
#						solver='saga')
lgr = LogisticRegression()
print('##################################################')
print('lgr done')
print('##################################################')
from sklearn.neural_network import BernoulliRBM

#rbm = BernoulliRBM(random_state = 1234,
#					verbose = True,
#					n_components = 2,
#					learning_rate = 0.06,
#					n_iter = 20
#)
rbm = BernoulliRBM()
print('##################################################')
print('rbm done')
print('##################################################')
#scores = model_selection.cross_val_score(clf,df[like_id].,df[gender],cv=10,scoring='accuracy')

from sklearn.pipeline import Pipeline


#for c in c_range:

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', lgr)])
print('##################################################')
print('main classifier done')
print('##################################################')

parameters = {
	"rbm__learning_rate": [0.1, 0.01, 0.001],
	"rbm__n_iter": [20, 40, 80],
	"rbm__n_components": [50, 100, 200],
	"logistic__C": [0.1, 1, 10.0, 100.0]
}

start = time.time()
grid_search = GridSearchCV(classifier, parameters, n_jobs = 8, verbose = 1)
grid_search.fit(X_train, y_train)

print('\ndone in %0.3fs' %(time.time() - start))
print('best score: %0.3f' %(gs.best_score_))

print('rbm + lgr done')
bestParameters = gs.best_estimator_.get_params()

for p in sorted(parameters.keys()):
	print('\t {:d}: {:d}'.format(p, bestParameters[p]))
#scores = model_selection.cross_val_score(classifier, X, y, cv=10, scoring='accuracy')
#print('avg: ' + str(numpy.average(scores)))
#print('max: ' + str(numpy.amax(scores)))

#import matplotlib.pyplot as plt
#plt.plot(c_range, c_scores)
#plt.xlabel('Value of C for LogReg')
#plt.ylabel('Cross-Validated Accuracy')


#classifier.fit(X,y)
#print('##################################################')
#print('model fit')
#print('##################################################')

#X_test = cv.transform(test[like_id])
#y_test = test[gender]
#y_predicted = clf.predict(X_test)
#print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
#X_train, X_test, y_train, y_test = train_test_split(df_like, df.gender, test_size=0.2, random_state=0)


#dump(classifier,"gender_rbm2lgrpipeline_jake.joblib")
#dump(cv,'gender_rbm2lgrpipeline_jake_cv.joblib')



#alpha = 0.5, class_prior=None

#clf = MultinomialNB()
#clf.fit(X_train, y_train)


#y_predicted = clf.predict(X_test)

#print("Accuracy: %.2f" % clf.score(y_test,y_predicted))

#classes = ['Male','Female']
#cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
#print("Confusion matrix:")
#print(cnf_matrix)