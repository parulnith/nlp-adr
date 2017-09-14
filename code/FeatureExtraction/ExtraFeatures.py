#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:55:57 2017

@author: Samuele Garda
"""
import re
import logging
from sklearn.feature_extraction import DictVectorizer
from collections import Counter    


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


def getPoSFeatures(pos_tweets):
  """
  Return PoS feature matrix. 
  Procedure:
    - create standard vector : set of PoS in data set
    - take frequencies
    - normalize (set to 0 non present feature)
  
  :param pos_tweets: tweets in form of PoS tag, e.g. [VB,DET,...] 
  :type pos_tweets: list of lists
  :rtype: numpy.ndarray
  """
  
  vec = DictVectorizer()
 
  pos_set = set()
  for pos_of_tweet in pos_tweets:
    pos_set.update(pos_of_tweet)
        
  dicted_pos_tweets  = [dict(Counter(pos_of_tweet)) for pos_of_tweet in pos_tweets]
    
  for pos in pos_set:
    for dicted_tweet in dicted_pos_tweets:
      if pos not in dicted_tweet.keys():
        dicted_tweet[pos] = 0
  
  for tweet in dicted_pos_tweets:
    assert len(tweet) == len(pos_set), "Noramlization was not successful"
    
  pos_features = vec.fit_transform(dicted_pos_tweets)
  
  logger.info("Created PoS feature matrix of shape : {}".format(pos_features.shape))
  
  return pos_features.toarray()

def getNegFeatures(tok_tweets):
  """
  Return negated words feature matrix. 
  Negated words are: words ending in `n`t` and `not`,`no`,`nobody`,`nothing`,`none`,`nowhere`,`neither`
  Procedure:
    - create standard vector : set of negated words
    - take frequencies
    - normalize (set to 0 non present feature)
  
  :param tok_tweets: tokenized tweets
  :type tok_tweets: list of lists
  :rtype: numpy.ndarray
  """
  
  vec = DictVectorizer()
    
  neg_words = set()
  dicted_neg_tweets = []
  neg_list = ['not', 'no', 'never', 'nobody', 'nothing', 'none', 'nowhere', 'neither']  
    
  for tweet in tok_tweets:
    neg_tweet = []
    for idx,tok in enumerate(tweet):
      if tok.endswith("n't") or tok in neg_list:
        next_idx = idx+1
        prev_idx = idx-1
        if next_idx < len(tweet):
          next_neg = 'not_'+tweet[next_idx]
          neg_tweet.append(next_neg)
          neg_words.add(next_neg)
        if prev_idx > 0:
          prev_neg = 'not_'+tweet[prev_idx]
          neg_tweet.append(prev_neg)
          neg_words.add(prev_neg)
    dicted_neg_tweets.append(dict(Counter(neg_tweet)))

  for neg in neg_words:
    for dicted_tweet in dicted_neg_tweets:
      if neg not in dicted_tweet.keys():
        dicted_tweet[neg] = 0
        
  for tweet in dicted_neg_tweets:
    assert len(tweet) == len(neg_words), "Noramlization was not successful"
    
  neg_features = vec.fit_transform(dicted_neg_tweets)
  
  logger.info("Created negated words feature matrix of shape : {}".format(neg_features.shape))
  
  return neg_features.toarray()

def getSentFeatures(tok_tweets,pos_file,neg_file):
  """
  Return sentiment words feature matrix. 
  Procedure:
    - create standard vector : set of exact word match in lexicons
    - take frequencies
    - normalize (set to 0 non present feature)
  
  :param tok_tweets: tokenized tweets
  :type tok_tweets: list of lists
  :param pos_file: Opinion Lexicon. https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
  :type pos_file: txt
  :param neg_file: Opinion Lexicon. https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
  :type neg_file: txt
  :rtype: numpy.ndarray
  """
  
  vec = DictVectorizer()
  
  n = open(pos_file).readlines()
  p = open(neg_file).readlines()
  
  neg_vocab = [w.strip() for w in n if not w.startswith(';')]
  pos_vocab = [w.strip() for w in p if not w.startswith(';')]
  
  pos_neg = set()
  
  logger.info("Performing search trough sentiment lexicon for each tweet. This might take a while...")
  
  pos_neg_tweets = []
  for tweet in tok_tweets:
    pos_neg_tweet = []
    for tok in tweet:
      if tok in pos_vocab or tok in neg_vocab:
        pos_neg_tweet.append(tok)
        pos_neg.add(tok)
    
    pos_neg_tweets.append(dict(Counter(pos_neg_tweet)))
    
  for p_n in pos_neg:
    for p_s_dicted_tweet in pos_neg_tweets:
      if p_n not in p_s_dicted_tweet.keys():
        p_s_dicted_tweet[p_n] = 0
        
  for tweet in pos_neg_tweets:
    assert len(tweet) == len(pos_neg), "Noramlization was not successful"
    
  p_n_features = vec.fit_transform(pos_neg_tweets)
  
  logger.info("Created sentiment words feature matrix of shape : {}".format(p_n_features.shape))
  
  return p_n_features.toarray() 

def getADRlexFeatures(tweets, adr_lexicon):
  """
  Return ADR lexicon feature matrix. 
  Procedure:
    - create standard vector : set of regular expression (multiple ADR words) search
    - take frequencies
    - normalize (set to 0 non present feature)
  
  :param tok_tweets: raw text tweets
  :type tok_tweets: list of lists
  :param adr_lexicon: ADR lexicon
  :type adr_lexicon: txt (new line separated terms)
  :rtype: numpy.ndarray
  """
    
  vec = DictVectorizer()
  
  lex = open(adr_lexicon).readlines()
  
  adr_vocab = []
  for line in lex:
    adr_word = line.split('\t')[1].strip()
    if len(adr_word) > 6:
      adr_vocab.append(adr_word)
      
  adr_set = set()
  
  patterns_dict = {re.compile(adr_term) : adr_term for adr_term in adr_vocab}
  
  logger.info("Performing regular expression search for each tweet. This might take a while...")
  
  tweets_adr = []
  for tweet in tweets:
    tweet_adr = []
    for pattern in patterns_dict.keys():
      adrs = re.search(pattern,tweet)
      if adrs != None:
        adr_term = patterns_dict.get(pattern)
        tweet_adr.append(adr_term)
        adr_set.add(adr_term)
    
    tweets_adr.append(dict(Counter(tweet_adr)))
    
  for adr_term in adr_set:
    for dicted_tweet in tweets_adr:
      if adr_term not in dicted_tweet.keys():
        dicted_tweet[adr_term] = 0
  
  for tweet in tweets_adr:
     assert len(tweet) == len(adr_set), "Noramlization was not successful"
     
  
  adr_features = vec.fit_transform(tweets_adr)
  
  logger.info("Created ADR words feature matrix of shape : {}".format(adr_features.shape))
  
  return adr_features.toarray() 