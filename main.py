# -*- coding: utf-8 -*-
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import csv
# steps:
# 1. tokenize
# 2. N-grams (2-grams and 3-grams) 
# 3. select params using 5-fold
# 4. Train
# 5. classify
# 6. Maybe try word2vec if there is time

if __name__ == '__main__':
    with open('uci-news-aggregator.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        inputs = []
        targets = []
        for row in reader:
            inputs.append(row[1])
            targets.append(row[4])
        
    bigrams = [ngrams(word_tokenize(s),2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>') for s in inputs]
    print(list(bigrams[1]))
    
    