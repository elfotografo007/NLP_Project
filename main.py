# -*- coding: utf-8 -*-
import csv
from logistic_model import LogisticRegressionModel
from bayes_model import BayesModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# steps:
# 1. tokenize
# 2. N-grams (2-grams and 3-grams) 
# 3. select params using 5-fold
# 4. Train
# 5. classify
# 6. Maybe try word2vec if there is time

def train_logistic():
    labels = ['b', 't', 'e', 'm']
    training = joblib.load('training_set.soan')
    test = joblib.load('test_set.soan')
    l_training = joblib.load('l_training_set.soan')
    l_test = joblib.load('l_test_set.soan')
    inputs = training + test
    
    # bigrams countvectorizer
    lm = LogisticRegressionModel(inputs, CountVectorizer(ngram_range=(2,2)))
    lm.train(training, l_training)
    lm.save('logistic_bigram_countvectorizer.soan')
    y = lm.classify(test)
    print('Bigram CountVectorizer')
    print(lm.getMetrics(test, l_test))
    lm.printConfusion(test, l_test, labels)
    print("Accuracy: " + str(accuracy_score(l_test, y)))
     
    # trigrams countvectorizer
    lm = LogisticRegressionModel(inputs, CountVectorizer(ngram_range=(3,3)))
    lm.train(training, l_training)
    lm.save('logistic_trigram_countvectorizer.soan')
    y = lm.classify(test)
    print('trigram CountVectorizer')
    print(lm.getMetrics(test, l_test))
    lm.printConfusion(test, l_test, labels)
    print("Accuracy: " + str(accuracy_score(l_test, y)))
     
    # bigrams TF-IDF
    lm = LogisticRegressionModel(inputs, TfidfVectorizer(ngram_range=(2,2)))
    lm.train(training, l_training)
    lm.save('logistic_bigram_tfidf.soan')
    y = lm.classify(test)
    print('Bigram TF-IDF')
    print(lm.getMetrics(test, l_test))
    lm.printConfusion(test, l_test, labels)
    print("Accuracy: " + str(accuracy_score(l_test, y)))
    
    # trigrams TF-IDF
    lm = LogisticRegressionModel(inputs, TfidfVectorizer(ngram_range=(3,3)))
    lm.train(training, l_training)
    lm.save('logistic_trigram_tfidf.soan')
    y = lm.classify(test)
    print('trigram TF-IDF')
    print(lm.getMetrics(test, l_test))
    lm.printConfusion(test, l_test, labels)
    print("Accuracy: " + str(accuracy_score(l_test, y)))
    #lm = LogisticRegressionModel.load('logistic_cv.soan')

if __name__ == '__main__':
#     with open('uci-news-aggregator.csv', 'r', encoding='Latin-1') as csvfile:
#         reader = csv.reader(csvfile)
#         inputs = []
#         targets = []
#         for row in reader:
#             inputs.append(row[1])
#             targets.append(row[4])
#    inputs = inputs[1:]
#   targets = targets[1:]
    # Split dataset into Training and test
#    training, test, l_training, l_test = train_test_split(inputs, targets, test_size=0.2)
#     joblib.dump(training, 'training_set.soan')
#     joblib.dump(test, 'test_set.soan')
#     joblib.dump(l_training, 'l_training_set.soan')
#     joblib.dump(l_test, 'l_test_set.soan')
    
    training = joblib.load('training_set.soan')
    test = joblib.load('test_set.soan')
    l_training = joblib.load('l_training_set.soan')
    l_test = joblib.load('l_test_set.soan')
    inputs = training + test
    targets = l_test + l_training
    labels = ['b', 't', 'e', 'm']
    lm = BayesModel(inputs)
    
    lm.train(training, l_training)
    y = lm.classify(test)

    print(lm)
    print(lm.count_label_occurences(targets))
    lm.parameter_tuning(training, l_training)
    print("Accuracy Training" + str(lm.eval(training, l_training)))
    print("Accuracy Test" + str(lm.eval(test, l_test)))
    print(lm.getMetrics(test, l_test))
    lm.printConfusion(test, l_test, labels)
    print(lm.classify(['Today’s Top Supply Chain and Logistics News From WSJ']))
    print(lm.classify(['Facebook Users Were Unwitting Targets of Russia-Backed Scheme']))
    print(lm.classify(['Explaining Health Insurance Cost-Sharing Reductions']))
    print(lm.classify(["Nelly's Rape Accuser Says She Will NOT Testify, Wants to Drop the Case"]))
    print(lm.classify(["Beyonce and Jay-z are expecting twins"]))
    print(lm.classify(["Twitter extends its character limit to 240 words"]))
    print(lm.classify(["Ebola virus strikes again in the West"]))

          
          