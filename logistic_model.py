# -*- coding: utf-8 -*-
from model import Model
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

'''
Created on Oct 6, 2017

@author: elfotografo007
'''

class LogisticRegressionModel(Model):
    '''
    Logistic Regression Model
    '''
    
    def __init__(self, corpus, feature_extractor=CountVectorizer(ngram_range=(2,2))):
        self.__bigram_vectorizer = feature_extractor
        self.__bigrams = self.__bigram_vectorizer.fit_transform(corpus)
       

    def train(self, inputs, targets, **options):
        self.__label_encoder = LabelEncoder()
        self.__train_labels = self.__label_encoder.fit_transform(targets)
        X = self.__bigram_vectorizer.transform(inputs)
        self.__model = LogisticRegressionCV(cv=5, solver='lbfgs', max_iter=100, multi_class='ovr',n_jobs=4, refit=True).fit(X, self.__train_labels)
        #self.__model = LogisticRegression(solver='liblinear', max_iter=200, multi_class='ovr',n_jobs=4).fit(X, self.__train_labels)
        print('Best C parameters: ' + str(self.__model.C_))
    
    def classify(self, inputs):
        X = self.__bigram_vectorizer.transform(inputs)
        prediction = self.__model.predict(X)
        return self.__label_encoder.inverse_transform(prediction)
    
if __name__ == '__main__':
    X = ['Fed official says weak data caused by weather, should not slow taper', "Fed's Charles Plosser sees high bar for change in pace of tapering", 'US open: Stocks fall after Fed official hints at accelerated tapering', "Fed risks falling 'behind the curve', Charles Plosser says", "Fed's Plosser: Nasty Weather Has Curbed Job Growth", 'Plosser: Fed May Have to Accelerate Tapering Pace', "Fed's Plosser: Taper pace may be too slow", "Fed's Plosser expects US unemployment to fall to 6.2% by the end of 2014", 'US jobs growth last month hit by weather:Fed President Charles Plosser']
    y = ['b', 'b', 'b', 'm', 'b', 't', 'm', 'e', 'g']
    #clf = LogisticRegression(solver='sag', max_iter=100, multi_class='ovr').fit(X, y)
    lm = LogisticRegressionModel(X)
    lm.train(X, y)
    print(lm.classify(['Fed official says weak data caused by weather, should not slow andres']))
    
    
    
    
    