import os
import sys
import glob
import math
import json

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

    #print(cnt/(ni+1))
    #print(cnt)
    #print(len(obj['context']))
print("averate len=%d"%(alllen//poet_cnt) )
print("all poet num=%d"%(poet_cnt) )
print("all context num=%d"%(all_cnt) )
