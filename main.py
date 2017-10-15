# -*- coding: utf-8 -*-
import csv
from logistic_model import LogisticRegressionModel
from bayes_model import BayesModel
from sklearn.model_selection import train_test_split
# steps:
# 1. tokenize
# 2. N-grams (2-grams and 3-grams) 
# 3. select params using 5-fold
# 4. Train
# 5. classify
# 6. Maybe try word2vec if there is time

if __name__ == '__main__':
    with open('uci-news-aggregator.csv', 'r', encoding='Latin-1') as csvfile:
        reader = csv.reader(csvfile)
        inputs = []
        targets = []
        for row in reader:
            inputs.append(row[1])
            targets.append(row[4])
    # Split dataset into Training and test
    training, test, l_training, l_test = train_test_split(inputs, targets, test_size=0.2)
        
    lm = BayesModel(inputs)
     
    lm.train(training, l_training)
    y = lm.classify(test)

    print(lm)
    lm.parameter_tuning(training, l_training)
    print(lm.eval(test, l_test))
    print(lm.getMetrics(test, l_test))
    lm.printConfusion(test, l_test)
    print(lm.classify(['Today’s Top Supply Chain and Logistics News From WSJ']))
    print(lm.classify(['Facebook Users Were Unwitting Targets of Russia-Backed Scheme']))
    print(lm.classify(['Explaining Health Insurance Cost-Sharing Reductions']))
    print(lm.classify(["Nelly's Rape Accuser Says She Will NOT Testify, Wants to Drop the Case"]))
    print(lm.classify(["Beyonce and Jay-z are expecting twins"]))
    print(lm.classify(["Twitter extends its character limit to 240 words"]))
    print(lm.classify(["Ebola virus strikes again in the West"]))

          
          