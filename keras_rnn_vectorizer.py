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

import glob
import json
import sys
import pickle
import numpy as np
import msgpack
import msgpack_numpy as mn
mn.patch()
import MeCab
import plyvel
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
  sentences = []
  answers   = []
  print("importing data from algebra...")
  for dbi, name in enumerate(glob.glob('./algebra/*.key')[:50000]):
    pure_name = name.replace('.key', '')
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

  model = build_model(maxlen=maxlen, in_dim=256, out_dim=len(tagvec))
  for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)
    #sys.exit()
    MODEL_NAME = "./models/snapshot.%09d.model"%(iteration)
    model.save(MODEL_NAME)

def eval():
  INPUT_NAME = "./source/bocchan.txt"
  MODEL_NAME = "./models/%s.model"%(INPUT_NAME.split('/').pop())

  #path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
  #text = open(path).read().lower()
  text = open(INPUT_NAME).read()
  print('corpus length:', len(text))

  chars = sorted(list(set(text)))
  print('total chars:', len(chars))
  char_indices = dict((c, i) for i, c in enumerate(chars))
  indices_char = dict((i, c) for i, c in enumerate(chars))
  maxlen = 40
  step = 3
  sentences = []
  next_chars = []
  for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
  model = load_model(MODEL_NAME)

  #for diversity in [0.2, 0.5, 1.0, 1.2]:
  for diversity in [1.0, 1.2]:
    print()
    print('----- diversity:', diversity)
    generated = ''
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)
    #for i in range(400):
    for i in range(200):
      x = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
      preds = model.predict(x, verbose=0)[0]
      #next_index = sample(preds, diversity)
      next_index = dynast(preds, diversity)
      next_char = indices_char[next_index]
      generated += next_char
      sentence = sentence[1:] + next_char
      sys.stdout.write(next_char)
      sys.stdout.flush()
    print()


def main():
  if '--preexe' in sys.argv:
     preexe()
  if '--train' in sys.argv:
     train()
  if '--eval' in sys.argv:
     eval()
if __name__ == '__main__':
  main()
