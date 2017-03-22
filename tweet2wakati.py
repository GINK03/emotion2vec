import os
import sys
import math
import MeCab 
import glob
import json
import dill
import re
import pickle
filenum = len(glob.glob('./out/*'))
m = MeCab.Tagger('-Owakati')
def emoji_process():
  emoji_freq = {}
  for ni, name in enumerate(glob.glob('./out/*')):
    if ni%1000 == 0:
      print('now iter {} {}'.format(ni, filenum), file=sys.stderr)
    raw = open(name, 'r').read()
    try:
      obj = json.loads(raw)
    except json.decoder.JSONDecodeError as e:
      continue
    emoji = re.compile(u'['
             u'\U0001F300-\U0001F64F'
             u'\U0001F680-\U0001F6FF'
             u'\u2600-\u26FF\u2700-\u27BF]+', 
             re.UNICODE)
    match = re.search(emoji, obj['txt']) 
    if match is None:
      continue
    for emoji in re.findall(emoji, obj['txt']):
      if emoji_freq.get(emoji) is None : emoji_freq[emoji] = 0
      emoji_freq[emoji] += 1
  open('emoji_freq.pkl', 'wb').write(pickle.dumps(emoji_freq))
  for emoji, freq in sorted(emoji_freq.items(), key=lambda x:x[1]*-1)[:2048]:
    print(emoji, freq)

def make_vectorize():
  term_vec    = pickle.loads(open('./term_vec.pkl', 'rb').read())
  emoji_index = {}
  emojis      = set()
  text_vec    = []
  for emoji, freq in sorted(filter(lambda x:x[1] > 5, pickle.loads(open('./emoji_freq.pkl', 'rb').read()).items()), key=lambda x:x[1]*-1)[:2048]:
    #print(emoji, freq)
    emoji_index[emoji] = len(emoji_index)
  index_emoji = {index:emoji for emoji, index in emoji_index.items()}
  [emojis.add(emoji) for emoji, index in emoji_index.items()]
  for ni, name in enumerate(glob.glob('./out/*')):
    if ni%1000 == 0:
      print('now iter {} {}'.format(ni, filenum), file=sys.stderr)
      pass
    if ni > 2000000: break
    f = open(name, 'r')
    raw = f.read()
    f.close()
    try:
      obj = json.loads(raw)
    except json.decoder.JSONDecodeError as e:
      continue
    emoji = re.compile(u'['
      u'\U0001F300-\U0001F64F'
      u'\U0001F680-\U0001F6FF'
      u'\u2600-\u26FF\u2700-\u27BF]', 
      re.UNICODE)
    match = re.search(emoji, obj['txt'])
    if match is None:
      continue
    vec = [0.]*2048
    for aemoji in re.findall(emoji, obj['txt']):
      if aemoji in emojis:
        vec[emoji_index[aemoji]] = 1.
    no_emoji_text = re.sub(emoji, '', obj['txt']) 
    buff = []
    mask = ['*']*30
    for i, term in enumerate(m.parse(no_emoji_text).strip().split()[:30]):
      mask[i] = term
    for term in mask:
      if term_vec.get(term) is not None:
        buff.append(term_vec[term])
      else:
        buff.append(term_vec['*'])
    text_vec.append( (buff, vec) )
    continue
  open('emojis.pkl', 'wb').write(pickle.dumps(emojis) )
  open('emoji_index.pkl', 'wb').write(pickle.dumps(emoji_index) )
  open('index_emoji.pkl', 'wb').write(pickle.dumps(index_emoji) )
  open('text_vec.pkl', 'wb').write(pickle.dumps(text_vec) )


def cal_maxlen():
  maxlen = 0
  num_freq = {}
  for ni, name in enumerate(glob.glob('./out/*')):
    if ni%1000 == 0:
      print('now iter {} {}'.format(ni, filenum), file=sys.stderr)
      pass
    raw = open(name, 'r').read()
    try:
      obj = json.loads(raw)
    except json.decoder.JSONDecodeError as e:
      continue
    wakati = m.parse(obj['txt']).strip().split()
    if num_freq.get(len(wakati)) is None: num_freq[len(wakati)] = 0
    num_freq[len(wakati)] += 1
  
  for num, freq in sorted(num_freq.items(), key=lambda x:x[0]):
    print( "{} {}".format(num, freq))
  open('maxlen.txt', 'w').write(str(max(num_freq.keys())))
  # 214個がmaxだった
  # なんか30語で十分だわ
def build_wakati_texts():
  maxlen = 25
  padding = 5
  term_freq = {}
  for ni, name in enumerate(glob.glob('./out/*')):
    buff = ['*']*30
    padding = ['*']*4
    if ni%10 == 0:
      print('now iter {} {}'.format(ni, filenum), file=sys.stderr)
    raw = open(name, 'r').read()
    try:
      obj = json.loads(raw)
    except json.decoder.JSONDecodeError as e:
      continue
    wakati = m.parse(obj['txt']).strip().split()
    for i, term in enumerate(wakati):
      try:
        buff[i] = term
      except IndexError as e:
        break
    padding.extend(buff)
    print(' '.join(padding) )

def vectorizer():
  term_vec = {}
  with open('./model.vec', 'r') as f:
    next(f)
    for i, line in enumerate(f):
      if i%1000 == 0:
        print("now iter %d "%(i))
      ents = iter(line.strip().split())
      term = next(ents)
      vec  = list(map(float, ents))
      term_vec[term] = vec 

  open('term_vec.pkl', 'wb').write(pickle.dumps(term_vec) )
      
if __name__ == '__main__':
  if '--emoji_process' in sys.argv:
    emoji_process()
  if '--make_vectorize' in sys.argv:
    make_vectorize()
  #cal_maxlen()
  #build_wakati_texts()
  #vectorizer()
  
