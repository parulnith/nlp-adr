#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:13:42 2017

@author: Samuele Garda
"""
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import	MultinomialNB
from sklearn.model_selection import StratifiedKFold,cross_validate
from gensim.parsing.porter import PorterStemmer
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.utils.class_weight import compute_class_weight
from collections import OrderedDict

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

def load_data(tweets_tsv, tweets_postag):
  """
  Return tweets id,user id,tweets label,raw tweets,tokenized tweets,
  tweets in PoS, PoS tagged tweets and stemmed tweets in a pandas Dataframe.
  
  :param tweets_tsv: <SID><tab><UID><tab><CLASS><tab><TWITTER_MESSAGE>
  :parm tweets_postag: ark-TweetNLP `./runTagger.sh --output-format conll --input-formt txt --input-field 4` 
  :rtype: pandas.DataFrame
  """
  
  o = open(tweets_tsv,'r',encoding = 'utf-8').readlines()
  p = open(tweets_postag).read()
  
  raw = p.split('\n\n')
  raw_pos_data = [line.split('\n') for line in raw]
  pos_data = []
  for tweet in raw_pos_data:
    pos_data.append([tuple(word_pos.split('\t')) for word_pos in tweet])
  
  stemmer = PorterStemmer()
  
  data = {}
  for idx,line in enumerate(o):
    tweet_id,user_id,adr,text = line.split('\t')
    data[tweet_id] = {}
    data[tweet_id]['user_id'] = user_id
    data[tweet_id]['adr'] = adr
    data[tweet_id]['raw_text'] = text
    data[tweet_id]['stem_text'] = [stemmer.stem(w_pos[0]) for w_pos in pos_data[idx]]
    data[tweet_id]['tok_text'] = [w_pos[0] for w_pos in pos_data[idx]]
    data[tweet_id]['pos_token'] = [w_pos[1] for w_pos in pos_data[idx]]
    data[tweet_id]['pos_text'] =  ['#'.join(list(w_pos)) for w_pos in pos_data[idx] ]
    
  df = pd.DataFrame.from_dict(data, orient='index')

  df.adr = df.adr.astype('int')
  df.user_id = df.user_id.astype('int')
  
  logger.info("Loaded dataframe from {0} and {1}".format(tweets_tsv,tweets_postag))
  logger.info("Dataframe information:\n")
  df.info()
  
  return df


class ValidationUtilities(object):
  """
  Wrapper class for validation utilities
  """

  
  def get_baseline_MNB(raw_tweets):
    """
    Generate BOW word frequency matrix X. Instatiate Naive Bayes classifier clf.
    This can be taken as a baseline classifier for the task.
    
    :param raw_tweets: unprocessed tweets
    :type raw_tweets: list of lists
    :rtype X: numpy.ndarray
    :rtype clf: sklearn classifier
    """
    
    logger.info("Creating baseline classifier (Naive Bayes)...")
    
    count_vectorizer = CountVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = (1,1), analyzer = 'word')
    clf =  MultinomialNB()
    
    X = count_vectorizer.fit_transform(raw_tweets)
    
    return clf,X.toarray()
  
  def get_class_weights(y):
    """
    Return class weights.
    
    :math:`N_{samples} / (N_{classes} * np.bincount(y))`
    
    :param y: vector of class labels
    :type y: numpy.ndarray
    :rtype: dict
    """
    
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    
    logger.info("Computed class weights. NON ADR : %s, ADR : %s" %(class_weights[0], class_weights[1]))
    
    class_weights_dict = {0 : class_weights[0], 1 : class_weights[1]}
    
    return class_weights_dict
  
  def standard_validation(clf,X,y):
    """
    Return macro averaged F1 score,precision,recall 
    for 10 fold stratified cross validation.
    
    :param clf: sklearn classifier
    :type clf: sklearn classifier
    :param X: feature matrix 
    :type X: numpy.ndarray
    :param y: class labels vector
    :type y: numpy.ndarray
    :rtype: OrderedDict
    """
    
    k_fold = StratifiedKFold(n_splits= 10, shuffle=True)
      
    measures = ['f1_macro','precision_macro','recall_macro']
    
    score = cross_validate(clf, X, y, cv= k_fold, n_jobs=1, scoring= measures)
  
    scored_dict = OrderedDict((k,np.mean(v)) for (k,v) in score.items())
    
    return scored_dict
  
  def compile_MLP(input_dim):
    """
    Return a compiled NN model. Building function for KerasClassifier (scikit-learn wrapper).
    Model description:
      - dense layer : 100 hidden units, ReLU activation
      - classification layer : 2 classes, softmax activation
      - loss function : categorical cross entropy
      - optimizer : adam
    
    :param input_dim: number of features of feature matrix
    :type input_dim: int
    :rtype: keras.models.Sequential
    
    """
    
    model = Sequential()
    model.add(Dense(100, input_dim = input_dim, activation = 'relu', name = "dense_layer"))
    model.add(Dense(2, activation="softmax", name = "classification_layer"))
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    return model
    
def concatenate(arrays):
  """
  Return concatenated arrays.
  
  :param arrays: arrays to be concatenated
  :type arrays: list
  :rtype: numpy.ndarray
  """
  
  concatenated = np.concatenate(arrays, axis = 1)
  
  return concatenated




    

