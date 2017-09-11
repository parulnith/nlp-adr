#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:06:03 2017

@author: Samuele Garda
"""
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.doc2vec import LabeledSentence
from gensim.models import doc2vec


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')


class LabeledLineSentence(object):
  """
  Iterator of tagged documents to be fed to gensim Doc2Vec.
  """
  def __init__(self, documents):
    """
    Generate iterator of tagged documents.
    :param documents: tokenized documents to be labeled
    :type documents: list of lists
    """
    self.documents = documents
  def __iter__(self):
    """
    Iterator of documents. Assign unique tag number to each document.
    :rtype: gensim.models.doc2vec.LabeledSentence
    """
    for uid, line in enumerate(self.documents):
      yield LabeledSentence(self.documents[uid],['SENT_%s' % uid])

def getLSAVectors(tok_tweets):
  """
  Return document-term matrix rank-lowered via Truncated SVD. Shape of the matrix is :math:`(D,K)`,
  where `D` is the number of documents and `K` the number of latent dimensions (300 in this case).
  Decompose original matrix in: :math:`U_{n x m}, \\Sigma_{m x m}, V_{m x n}`. 
  Reconstruct vector space with latent dimension: :math:`U_{nxk} \cdot \\Sigma_{k}`.
  Model description:
    - ngrams : 1,2,3
    - tf-idf weighting : :math:`tfidf(t,d) = \\frac{freq_{t,d}}{{\sum_{t' \in d}}{freq_{t',d}}}`
    - latent dimension : 300
  
  :param tok_tweets: tokenized tweets
  :type tok_tweets: list of lists
  :rtype: numpy.ndarray
  """
    
  tfidf_vec = TfidfVectorizer(preprocessor = ' '.join, tokenizer = str.split, ngram_range = (1,3), analyzer = 'word')
  
  X_tfidf = tfidf_vec.fit_transform(tok_tweets)
    
  svd = TruncatedSVD(n_components = 300, random_state = 7)
  
  logger.info("\n\nApplying SVD to the processed matrix of shape {}".format(X_tfidf.shape))
  
  denseLSAvecs = svd.fit_transform(X_tfidf)
  
  logger.info("Created LSA matrix of shape : {}\n\n".format(denseLSAvecs.shape))
  
  return denseLSAvecs
  
  
def getDoc2VecVectors(tok_tweets):
  """
  Return document embeddings matrix. Shape of the matrix is :math:`(D,K)`,
  where `D` is the number of documents and `K` the size of the embeddings (300 in this case).
  Model description
    - dm = 1 : PV-DM
    - hs = 0 : negative sampling
    - negative = 50 : number of words extracted in negative sampling
    - dm_mean = 1 : taking mean of word vectors and paragraph vector
    - window = 3 : sliding window size
    - size = 300 : dimension of embeddings
   
  :param tok_tweets: tokenized tweets
  :type tok_tweets: list of lists
  :rtype: numpy.ndarray
  """
    
  tagged_docs = LabeledLineSentence(tok_tweets)
  
  model_pv_dm = doc2vec.Doc2Vec(dm = 1, window = 3, size = 300, hs = 0 , min_count = 1, dm_concat = 0,
                                dm_mean = 1, workers = 1,negative = 50, seed = 7)
    
  model_pv_dm.build_vocab(tagged_docs)
  
  corpus = list(tagged_docs)
  
  logger.info("\n\nTraining Paragraph Vector-Distributed Memory model...")
 
  model_pv_dm.train(corpus,total_examples = len(corpus),epochs = 20)
  
      
  d2v = np.array([model_pv_dm.docvecs[i] for i in range(len(corpus))])
  
  logger.info("Created paragraph2vec matrix of shape : {}\n\n".format(d2v.shape))
  
  return d2v
  
  
  