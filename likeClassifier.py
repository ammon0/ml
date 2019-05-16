import pandas
from sklearn import metrics
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
TESTING =False

def likeLogReg(profileTable, relationTable):
    #print(profileTable)
    #print(relationTable)
    relationTable['like_id'] = relationTable['like_id'].astype(str)
    relationTable = relationTable.groupby('userid',as_index=False).agg({'like_id': lambda x: "%s" % ' '.join(x)})
    #relationTable.rename(columns = {'':'like_id'}, inplace=True)
    #relationTable = relationTable.rename ['userid', 'like_id']
    #df = pandas.merge(profileTable, relationTable, on='userid',how='left')
    #print('1')
    #print(relationTable)
    cv = load('/home/itadmin/ml/likes/age_likes_lgr_jake_cv.joblib')
    #cv = CountVectorizer()
    results = pandas.DataFrame(index=relationTable['userid'])
    X = cv.transform(relationTable['like_id'])
    #print('2')
    #print(X)
    #results = pandas.DataFrame(index=profileTable['userid'])
    #print(results)    
    likeLogisticRegression = load('/home/itadmin/ml/likes/age_likes_lgr_jake.joblib')
    
    results['age'] = likeLogisticRegression.predict(X).astype(int)
    #print('3')
    #print(results)
    if TESTING:
        copy = profileTable.set_index('userid')
        
        copy.sort_index(inplace=True)
		
        results.sort_index(inplace=True)
        print(copy)
        print(results)
        print(metrics.accuracy_score(copy['age'],results['age']))
    return results

