from __future__ import print_function
from keras.models               import Sequential, load_model
from keras.layers               import Dense, Activation
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

def build_model(maxlen=None, out_dim=None, in_dim=256):
  print('Build model...')
  model = Sequential()
  model.add(GRU(128*5, return_sequences=True, input_shape=(maxlen, in_dim)))
  model.add(BN())
  model.add(GN(0.2))
  model.add(GRU(128*5, return_sequences=False, input_shape=(maxlen, in_dim)))
  #model.add(BN())
  model.add(GN(0.2))
  model.add(Dense(out_dim))
  model.add(Activation('linear'))
  model.add(Activation('sigmoid'))
  optimizer = Adam()
  model.compile(loss='binary_crossentropy', optimizer=optimizer) 
  return model


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def dynast(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(preds)

def preexe():
  # タグのベクタライズ
  # 1024個に限定
  tag_freq = {}
  for gi, name in enumerate(glob.glob('./contents/*')):
    if gi%500 == 0:
      print("now tag-building iter %d"%gi)
    data = open(name, 'r').read()
    try:
      obj = json.loads(data)
    except:
      continue
    tags = obj['tags']
    for tag in tags:
      if tag_freq.get(tag) is None: tag_freq[tag] = 0
      tag_freq[tag] += 1
  tag_index = {}
  for tfi, (tag, freq) in enumerate(sorted(tag_freq.items(), key=lambda x:x[1]*-1)[:1024]):
    tag_index[tag] = tfi
    print(tag, tfi)
  # タグデータの保存
  open('tag_index.pkl', 'wb').write(pickle.dumps(tag_index))
  term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
  # 分かち書きと量子化
  m = MeCab.Tagger('-Owakati')
  for gi, name in enumerate(glob.glob('./contents/*')):
    if gi%50 == 0:
      print("now kvs-generating iter %d"%gi)
    with open(name, 'r') as f:
      data = f.read()
    try:
      obj = json.loads(data)
    except:
      continue
    if obj['context'] is None:
      continue
    tags = obj['tags']
    try:
      terms = m.parse(obj['context']).split()
    except AttributeError as e:
      continue
    contexts = []
    for term in terms[:400]:
      try:
        contexts.append(term_vec[term]) 
      except KeyError as e:
        contexts.append(term_vec["ダミー"])
    #print(context)
    tagvec = [0.]*len(tag_index)
    for tag in tags:
      try:
        tagvec[tag_index[tag]] = 1.
      except:
        # 見つからなかったtagは無視する
        pass
    tagvec_enc = msgpack.packb(tagvec, default=mn.encode)
    last_name = name.split('/')[-1]
    open('algebra/%s.key'%last_name, 'wb').write(tagvec_enc)
    fasttext_enc = msgpack.packb(np.array(contexts), default=mn.encode)
    open('algebra/%s.value'%last_name, 'wb').write(fasttext_enc)

def train():
  maxlen = 200
  step = 1
  print("importing data from algebra...")
  model = build_model(maxlen=maxlen, in_dim=256, out_dim=1024)
  for ci, slicer in enumerate(Cycle(list(range(0,10)))):
    print("iter ci=%d, slicer=%d"%(ci, slicer))
    sentences = []
    answers   = []
    for dbi, name in enumerate(glob.glob('./algebra/*.key')[5000*slicer:5000*slicer + 5000]):
      pure_name = name.replace('.key', '')
      if dbi%500 == 0:
        print("now loading db iter %d"%dbi)
      tagvec = msgpack.unpackb(open('%s.key'%pure_name, 'rb').read(), object_hook=mn.decode)
      fasttext = msgpack.unpackb(open('%s.value'%pure_name, 'rb').read(), object_hook=mn.decode)
      sentences.append(fasttext)
      answers.append(tagvec)
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, 256), dtype=np.bool)
    y = np.zeros((len(sentences), len(tagvec)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
      for t, term_vec in enumerate(sentence[:maxlen]):
        X[i, t, :] = term_vec
      y[i, :] = answers[i]
    
    for iteration in range(1, 4):
      print()
      print('-' * 50)
      print('Iteration', iteration)
      model.fit(X, y, batch_size=128, nb_epoch=1)
    MODEL_NAME = "./models/snapshot.%09d.model"%(ci)
    model.save(MODEL_NAME)
    del X
    del y
    if ci > 100:
      break

def pred():
  model  = load_model(sorted(glob.glob('./*.model'))[-1] )
  for name in glob.glob('./samples/*'):
    text = open(name).read() * 10
    #print(text)
    tag_index = pickle.loads(open('tag_index.pkl', 'rb').read())
    term_vec = pickle.loads(open('term_vec.pkl', 'rb').read())
    m = MeCab.Tagger('-Owakati')
    terms = m.parse(text).split()
    contexts = []
    for term in terms[:200]:
      try:
        contexts.append(term_vec[term]) 
      except KeyError as e:
        contexts.append(term_vec["ダミー"])
    result = model.predict(np.array([contexts]))
    result = {i:w for i,w in enumerate(result.tolist()[0])}
    for tag, index in sorted(tag_index.items(), key=lambda x:result[x[1]]):
      print(name, tag, result[index], index)
def main():
  if '--preexe' in sys.argv:
     preexe()
  if '--train' in sys.argv:
     train()
  if '--pred' in sys.argv:
     pred()
if __name__ == '__main__':
  main()
