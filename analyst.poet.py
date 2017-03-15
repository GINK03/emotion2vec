import os
import sys
import glob
import math
import json
import MeCab
from pathlib import Path 
import pickle
def checker():
  poet_cnt = 0
  all_cnt  = 0
  alllen   = 0
  for ni, name in enumerate(glob.glob('./contents/*')):
    all_cnt += 1
    try:
      obj = json.loads(open(name).read())
    except json.decoder.JSONDecodeError as e:
      continue
    if 'ポエム' in obj['context'] or 'ポエム' in obj['tags']:
      poet_cnt += 1
      alllen += len(obj['context'])
  print("averate len=%d"%(alllen//poet_cnt) )
  print("all poet num=%d"%(poet_cnt) )
  print("all context num=%d"%(all_cnt) )

def dump():
  name = Path("./model.bin")
  if not name.is_file():
    m = MeCab.Tagger('-Owakati')
    with open('dump.txt', 'w') as f:
      for ni, name in enumerate(glob.glob('./contents/*')):
        try:
          obj = json.loads(open(name).read())
        except json.decoder.JSONDecodeError as e:
          continue
        try:
          f.write(m.parse(obj['context']))
          f.write('\n')
        except TypeError as e:
          continue
    os.system("./fasttext skipgram -dim 256 -minCount 1 -input dump.txt  -output model")
  term_vec = {}
  with open('model.vec', 'r') as f:
    next(f)
    for line in f:
      ents = line.split()
      term = ' '.join(ents[:-256])
      vec  = list(map(float, ents[-256:]))
      term_vec[term] = vec
    open('term_vec.pkl', 'wb').write(pickle.dumps(term_vec))
if __name__ == '__main__':
  if '--check' in sys.argv:
    checker()
  if '--dump' in sys.argv:
    dump()
