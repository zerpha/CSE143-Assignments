# -*- coding: utf-8 -*-
"""Ass2Part3Try2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O_BGA-PhfTBIXLppS3k1BvipwfE5mBO6
"""

# Keras
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, GRU, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.datasets import imdb
from keras.preprocessing import sequence

## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.manifold import TSNE

vocabulary_size = 5000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 256

vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
test_x = np.concatenate((X_train, X_test), axis=0)
test_y = np.concatenate((y_train, y_test), axis=0)
train_x = data[10000:] #40000
train_y = targets[10000:] #40000
dev_x = train_x[:10000] #10000
dev_y = train_y[:10000] #10000

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
dev_x = sequence.pad_sequences(dev_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)
print('train_x shape:', train_x.shape)
print('dev_x shape:', dev_x.shape)
print('test_x shape:', test_x.shape)

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

word2id = imdb.get_word_index()   # dictionary from words to integers (the id of the word in the vocab)
id2word = {i: word for word, i in word2id.items()}
embedding_matrix = np.zeros((vocabulary_size, 100))
for word, index in word2id.items():
   # print(word2id)
   if index > vocabulary_size - 1:
       continue
   else:
       embedding_vector = embeddings_index.get(word)
       # print("embedding")
       # print(embedding_vector)
       if embedding_vector is not None:
           embedding_matrix[index] = embedding_vector

print(len(embedding_matrix))
for i in range(0, 3):
   print("The glove embedding for '{}' is {} ".format(list(word2id.keys())[i], embedding_matrix[i]))

from keras.optimizers import RMSprop
model_glove = tf.keras.Sequential()
model_glove.add(tf.keras.layers.Embedding(vocabulary_size, 100, input_length=80, weights=[embedding_matrix], trainable=False))
model_glove.add(tf.keras.layers.Dropout(0.2))
model_glove.add(tf.keras.layers.GRU(256, return_sequences=True))
model_glove.add(tf.keras.layers.GRU(256))
model_glove.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer = keras.optimizers.RMSprop(learning_rate=0.0005)
    , metrics=['accuracy'])

model_glove.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=5,
          validation_data= (dev_x, dev_y))

score, acc = model_glove.evaluate(test_x, test_y,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)