#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:47:39 2017

@author: Samuele Garda
"""

"""
CNN class with helper method to train the model with pre-computed word2vec word embeddings.
"""
import numpy as np
import logging
from collections import defaultdict, Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers.core import Dense,Dropout
from keras.layers.convolutional import Conv1D,MaxPooling1D,AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Input
from keras.models import Model
from keras.layers.merge import Concatenate
from keras import backend as K

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')

  
class CNN(object):
  """
  A Convolutional Neural Network to learn compositionality of pre-trained word embeddings with
  1 dimensional convolutions. The main methods of this object are : `train_CNN` which trains the model 
  as a classification algorithm and `getCNNfeatures` which, after training, computes feature 
  representation for each document in the input matrix.
  """
  
  def __init__(self, embeddings_dim = 150, vocab_size = 4000, num_filters = 32, 
               num_words = (2,3), batch_size = 64, num_epochs = 1):
    """
    Contstruct new CNN feature extractor.
    
    :param embeddings_dim: dimesinoality of pre-trained word embeddings
    :type embeddings_dim: int
    :param vocab_size: most frequent words to be retained in constructing the embeddings matrix
    :type vocab_size: int
    :param num_filters: number of feature maps to be extracted for each convolutional layer
    :param num_words: region size of the convolution
    :type num_words: tuple
    :param batch_size: size of batches to train the model with
    :type batch_size: int
    :param num_epochs: number of epochs for training
    :type num_epochs: int
    
    :rtype: CNN
    """
    self.embeddings_dim = embeddings_dim
    self.vocab_size = vocab_size
    self.num_filters = num_filters
    self.num_words = num_words
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.embeddings_vocab_size = vocab_size + 1
    self.embeddings_dict = None
    self.word2index = defaultdict(int)
    self.index2word = dict
    self.X = None
    self.Y  = None
    self.cnn_feature = None
    self.maxlen = 0
    self.embedding_matrix = None
    self.base_model = None
    
    
  def _create_embeddings_dict(self,we_file):
    """
    Creates word embeddings dictionary extracted from file.
    
    :param we_file: embeddings file in word2vec format  
    """
    f = open(we_file).readlines()
    f.remove(f[0])
    self.embeddings_dict = {}
    for line in f:
      line = line.replace('\\','')
      values = line.split()
      try:
        word = values[0]
        coef = np.asarray(values[1:], dtype = 'float32')
        self.embeddings_dict[word] = coef
      except ValueError:
        word = values[0]+' '+values[1]
        coef = np.asarray(values[2:], dtype = 'float32')
        self.embeddings_dict[word] = coef
    
    logger.info("Loaded {0} embeddings from file : {1}".format(len(self.embeddings_dict),we_file) )

  def _generate_input(self,text,y):
    """
    Create input data for CNN. 
    
    :param text: input documents to be transformed in sequences
    :type text: list of lists
    :param y: vector of class labels
    :type y: numpy.ndarray
    """
  
    counter = Counter()
    for tweet in text:
      words = [x.lower() for x in tweet]
      if len(words) > self.maxlen:
        self.maxlen = len(words)
      for word in words:
        counter[word] += 1
    
    for wid,word in enumerate(counter.most_common(self.vocab_size)):
      self.word2index[word[0]] = wid + 1
    self.index2word = {v : k for k,v in self.word2index.items()}
    
    logger.info("Created dictionary word2index of size (selected number of most frequent words): {0}".format(self.vocab_size))
    
    xs = []
    for tweet in text:
      words = [x.lower() for x in tweet]
      wids = [self.word2index[word] for word in words]
      xs.append(wids)
    
    self.X = pad_sequences(xs, maxlen = self.maxlen)
    self.Y = np_utils.to_categorical(y)
  
    logger.info("Created data set for CNN with X : {0} and Y : {1}".format(self.X.shape,self.Y.shape))

  def _create_embeddings_matrix(self):
    """
    Create the word embedding matrix (weights of first layer). Set to vectors of 0s embeddings
    of words not retrieved in data set.
    """
    
    found = 0
    self.embedding_matrix = np.zeros((self.embeddings_vocab_size, self.embeddings_dim))
    for word, i in self.word2index.items():
      embedding_vector = self.embeddings_dict.get(word)
      if embedding_vector is not None:
          found += 1
          # words not found in embedding index will be all-zeros.
          self.embedding_matrix[i] = embedding_vector
      else:
        pass
    
    logger.info("Retrieved {} word embedding vectors".format(found))
    logger.info("Created embedding matrix for CNN of shape : {0}".format(self.embedding_matrix.shape))
  
  
  def train_CNN(self,we_file, text,y, model = 'cnn-static'):
    """
    Train CNN model for document classification.
    Model architecture:
      - input layer : emebddings matrix
      - dropout : 0.25  of the input units dropped
      - convolutional layers : one dimensional, region sizes = self.num_words, ReLU activation  
      - max pooling layer : pool size is 2 (average pooling if model is `sswe-static` or `sswe-non-static`)
      - flattening layer: transform output of pooling layer in fixed size vector
      - concatenate layer : concatenate flattened outputs
      - dropout : 0.50  of the input units dropped
      - classification layer : 2 classes, softmax activation
      - loss function : categorical cross entropy
      - optimizer : adam
    
    If the model type is set to `ǹon-static` the embeddings weights are updated during the training process
    jointly with the other parameters of the model.
    
    :param we_file: embeddings file in word2vec format
    :param text: input documents to be transformed in sequences
    :type text: list of lists
    :param y: vector of class labels
    :type y: numpy.ndarray
    :param model: options = [`cnn-static`,`cnn-non-static`,`sswe-static`,`sswe-non-static`]
    :type model: string
    """
    
    self._create_embeddings_dict(we_file)
    self._generate_input(text,y)
    self._create_embeddings_matrix()
    
    if model == 'cnn-static' or model == 'sswe-static':
      train_we = False
    elif model == 'cnn-non-static' or model == 'sswe-non-static':
      train_we = True
    
    main_input = Input(shape=(self.X.shape[1],), dtype='float32', name='main_input')
    
    
    embed = Embedding(self.embeddings_vocab_size, self.embeddings_dim, 
                                  input_length=self.maxlen,weights=[self.embedding_matrix],
                                  name = "embedding_layer",
                                  trainable = train_we)(main_input)
    
    dropout1 = Dropout(0.25)(embed)
    
    convolutional_filters = []
    for filter_size in self.num_words:
      
      conv = Conv1D(filters=self.num_filters, 
                           kernel_size= filter_size, activation="relu",
                           name = "convolution_layer_{}".format(filter_size))(dropout1)
      
      if model == 'sswe-static' or model == 'sswe-non-static':
        
        avg_pool = AveragePooling1D(name = "avg_pooling_layer_{}".format(filter_size))(conv)
        flatten_avg =  Flatten(name = "flatten_avg_layer_{}".format(filter_size))(avg_pool)
        convolutional_filters.append(flatten_avg)
        
        
      max_pool = MaxPooling1D(pool_size = 2, name = "max_pooling_layer_{}".format(filter_size))(conv)
      flatten_max  = Flatten(name = "flatten_max_layer_{}".format(filter_size))(max_pool)
      convolutional_filters.append(flatten_max)
      
    
    merged = Concatenate()(convolutional_filters)
    
    dropout2 = Dropout(0.5)(merged)
  
    final = Dense(2, activation="softmax", name = "softmax_layer")(dropout2)
    
    self.base_model = Model(inputs = main_input, outputs = final)
    
    logger.info(self.base_model.summary())
    
    self.base_model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    
    self.base_model.fit(self.X, self.Y, batch_size= self.batch_size,epochs= self.num_epochs)
    
  
  def getCNNfeautures(self,we_file,text,y,model = 'cnn-non-static'):
    """
    Return docmuent features extracted via the model trained with `train_CNN`. After the training is
    completed a model stopped at the penultimate layer is used generate the features of the input data set.
    The shape of the matrix is :math:`(D,K)`. `D` is the number of documents and :math:`K = P_{o} \cdot N_{fm}`, where :math:`P_{o}`
    is the output vector size of the pooling layer and :math:`N_{fm}` is the number of filters applied.
    
    :param we_file: embeddings file in word2vec format
    :param text: input documents to be transformed in sequences
    :type text: list of lists
    :param y: vector of class labels
    :type y: numpy.ndarray
    :param model: options = [`cnn-static`,`cnn-non-static`,`sswe-static`,`sswe-non-static`]
    :type model: string
    """
        
    self.train_CNN(we_file,text,y,model = model)
    
    out_layer = -2
    
    get_cnn_feature = K.function([self.base_model.layers[0].input, K.learning_phase()],
                                  [self.base_model.layers[out_layer].output])
    
    cnn_features = get_cnn_feature([self.X,0])[0]
    
    logger.info("Extracted CNN feature : {}".format(cnn_features.shape))
    
    return cnn_features