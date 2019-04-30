import pandas
from sklearn import metrics
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer

def likeLogReg(profileTable, relationTable):
    
    
    relationTable['like_id'] = relationTable['like_id'].astype(str)
    relationTable = relationTable.groupby('userid')['like_id'].apply(' '.join)
    df = pandas.merge(profileTable, relationTable, on='userid',how='left')
    cv = CountVectorizer(ngram_range = (1,5),max_features=102000)
    X = cv.fit_transform(df['like_id'])
    
    likeLogisticRegression = load('/home/itadmin/ml/likes/gender_LogisticRegression1_jake.joblib')
    return likeLogisticRegression.predict(X)
