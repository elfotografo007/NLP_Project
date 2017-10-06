# -*- coding: utf-8 -*-
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
        pass
    
    def printConfusion(self):
        pass
    
    def save(self):
        pass
    
    def load(self):
        pass