from __future__ import print_function
from keras.models               import Sequential, load_model
from keras.layers               import Input, Dense, Activation
from keras.layers               import LSTM, GRU, SimpleRNN
from keras.optimizers           import RMSprop, Adam
from keras.utils.data_utils     import get_file
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.noise         import GaussianDropout as GD
import numpy as np
import random
import sys
import tensorflow               as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
import glob
import json
import pickle
import msgpack
import msgpack_numpy as mn
mn.patch()
import MeCab
import plyvel
from itertools import cycle as Cycle
import dill

def discriminator_model():
  from keras.layers import Input, Dense, Embedding, merge, Convolution2D as Conv2D, MaxPooling2D, Dropout, ZeroPadding2D
  from keras.layers.core import Reshape, Flatten
  from keras.models import Model, load_model
  from keras.layers.merge import add, concatenate
  sequence_length    = 30
  embedding_dim      = 256 
  vocabulary_size    = 10
  num_filters        = 512
  filter_sizes       = [3,4,5,1,2]
  inputs = Input(shape=(sequence_length,embedding_dim,), dtype='float64')
  #embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
  reshape = Reshape((sequence_length,embedding_dim,1))(inputs)
  #Conv2D(512, (3, 256), padding="valid", kernel_initializer="normal", data_format="channels_last", activation="relu")

  conv_0      = Conv2D(num_filters, (filter_sizes[0], embedding_dim), kernel_initializer="normal", data_format="channels_last", activation="relu")(reshape)
  conv_1      = Conv2D(num_filters, (filter_sizes[1], embedding_dim), kernel_initializer="normal", data_format="channels_last", activation="relu")(reshape)
  conv_2      = Conv2D(num_filters, (filter_sizes[2], embedding_dim), kernel_initializer="normal", data_format="channels_last", activation="relu")(reshape)
  conv_3      = Conv2D(num_filters, (filter_sizes[3], embedding_dim), kernel_initializer="normal", data_format="channels_last", activation="relu")(reshape)
  conv_4      = Conv2D(num_filters, (filter_sizes[4], embedding_dim), kernel_initializer="normal", data_format="channels_last", activation="relu")(reshape)
  # `MaxPooling2D(pool_size=(28, 1), padding="valid", data_format="channels_last", strides=(1, 1))
  maxpool_0   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding="valid", data_format="channels_last")(conv_0)
  maxpool_1   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding="valid", data_format="channels_last")(conv_1)
  maxpool_2   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding="valid", data_format="channels_last")(conv_2)
  maxpool_3   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[3] + -3, 1), strides=(1,1), padding="valid", data_format="channels_last")(conv_3)
  maxpool_4   = MaxPooling2D(pool_size=(sequence_length - filter_sizes[4] + -2, 1), strides=(1,1),padding="valid", data_format="channels_last")(conv_4)

  merged = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)
  
  output = Dense(units=2048, activation='sigmoid')( \
             Activation('linear')( \
               Dropout(0.5)( \
                 Flatten()(merged) ) ) )

  model = Model(inputs=inputs, outputs=output)
  adam = Adam()
  model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

  return model

def main_train():
  print("importing data from serialized...")
  text_vec    = pickle.loads(open('./text_vec.pkl', 'rb').read())
  print("finished data from serialized...")

  print('Vectorization...')
  X = np.zeros((len(text_vec), 30, 256), dtype=np.float64)
  y = np.zeros((len(text_vec), 2048), dtype=np.float64)
  for i, (text, ans) in enumerate(text_vec):
    if i%10000 == 0:
      print("building training vector... iter %d"%i)
    for t, term in enumerate(text):
      X[i, t, :] = term
    y[i, :] = ans
  model = discriminator_model()
  for iteration in range(1, 101):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)
    MODEL_NAME = "./models/snapshot.%09d.model"%(iteration)
    model.save(MODEL_NAME)
  sys.exit()

def pred():
  m = MeCab.Tagger('-Owakati')
  emoji_index = pickle.loads(open('./emoji_index.pkl', 'rb').read())
  index_emoji = {index:emoji for emoji, index in emoji_index.items()}
  
  print('now loading term_vec.pkl...')
  term_vec    = pickle.loads(open('term_vec.pkl', 'rb').read())
  print('finished loading term_vec.pkl...')
  model_type = sorted(glob.glob('./models/*.model'))[-1]
  print("model type is %s"%model_type)
  model  = load_model(model_type)
  print("finished model type is %s"%model_type)
  for line in sys.stdin:
    line = line.strip()
    print(line, end=" ")
    buff = ["*"]*30
    for i,term in enumerate(m.parse(line).strip().split()[:30]):
      buff[i] = term
    X = []
    for term in buff:
      if term_vec.get(term) is not None:
        X.append(term_vec[term])
      else:
        X.append(term_vec["*"])
    results = model.predict(np.array([X]))
    res = {index_emoji[i]:score for i,score in enumerate(results[0].tolist())}
    for emoji, score in sorted(filter(lambda x:x[1]>0.01, res.items()), key=lambda x:x[1]*-1)[:20]:
      print(emoji, "%d"%(int(score*100)), end=" ")
    print()
def main():
  if '--train' in sys.argv:
     main_train()
  if '--pred' in sys.argv:
     pred()
if __name__ == '__main__':
  main()
