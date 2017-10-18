# -*- coding: utf-8 -*-
from model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


'''
Created on Oct 6, 2017

@author: elfotografo007
'''

class BayesModel(Model):
    '''
    Bayes Model
    '''
    def __init__(self, corpus):
        self.__bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), analyzer='word', stop_words='english')
        self.__bigrams = self.__bigram_vectorizer.fit_transform(corpus)
        print (self.__bigrams.shape)
        #print(self.__bigram_vectorizer.get_stop_words())

    def train(self, inputs, targets, **options):
        self.__label_encoder = LabelEncoder()
        self.__train_labels = self.__label_encoder.fit_transform(targets)
        X = self.__bigram_vectorizer.transform(inputs)
        self.__model = MultinomialNB(alpha=0.20000000000000001).fit(X, self.__train_labels)
            
        
    def parameter_tuning(self, inputs, targets, **options):
        self.__label_encoder = LabelEncoder()
        self.__train_labels = self.__label_encoder.fit_transform(targets)
        X = self.__bigram_vectorizer.transform(inputs)
        kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        evaluation = cross_val_score(self.__model, X, self.__train_labels, cv=kf)
        print(evaluation)
        rmses = [np.sqrt(np.absolute(mse)) for mse in evaluation]
        avg_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        print(avg_rmse)
        print(std_rmse)
        param_grid = {"alpha": np.array([1, 0.1, 0.01, 0.2, 0.02, 0.3, 0.03, 0.4, 0.04, 0.5, 0.05, 0.6, 0.06, 0.7, 0.07, 0.8, 0.08, 0.9, 0.09, 0])}
        gcv = GridSearchCV(self.__model, param_grid, cv=kf)
        gcv.fit(X, self.__train_labels)
        print("Best alpha parameter" + str(gcv.best_params_))
        print("Best alpha score" + str(gcv.best_score_))
        
    def classify(self, inputs):
         X = self.__bigram_vectorizer.transform(inputs)
         prediction = self.__model.predict(X)
         return self.__label_encoder.inverse_transform(prediction)

    def eval(self, inputs, targets):
        predicted = self.classify(inputs)
        return accuracy_score(targets, predicted)
    
if __name__ == '__main__':
    X = ['Fed official says weak data caused by weather, should not slow taper', "Fed's Charles Plosser sees high bar for change in pace of tapering", 'US open: Stocks fall after Fed official hints at accelerated tapering', "Fed risks falling 'behind the curve', Charles Plosser says", "Fed's Plosser: Nasty Weather Has Curbed Job Growth", 'Plosser: Fed May Have to Accelerate Tapering Pace', "Fed's Plosser: Taper pace may be too slow", "Fed's Plosser expects US unemployment to fall to 6.2% by the end of 2014", 'US jobs growth last month hit by weather:Fed President Charles Plosser']
    y = ['b', 'b', 'b', 'm', 'b', 't', 'm', 'e', 'g']
    #clf = LogisticRegression(solver='sag', max_iter=100, multi_class='ovr').fit(X, y)
    lm = BayesModel(X)
    lm.train(X, y)
    print(lm.classify(['Beyonce and Jay z are having another baby']))