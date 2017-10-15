# -*- coding: utf-8 -*-
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

'''
Created on Oct 6, 2017

@author: elfotografo007
'''

class Model(object):
    '''
    Almost abstract class to create Language Models
    '''


    def train(self, inputs, targets, **options):
        pass
    
    def classify(self, inputs):
        pass

    def getMetrics(self, inputs, targets):
        return classification_report(targets, self.classify(inputs))
    
    def printConfusion(self, inputs, targets):
        print(confusion_matrix(targets, self.classify(inputs)))
    
    def save(self, filename):
        joblib.dump(self, filename)
    
    @staticmethod
    def load(filename):
        return joblib.load(filename)
        
        