import json
import numpy as np
import re
result = []
with open("predict.txt") as f:
	
	for idx, l in enumerate(f):
		l = l.strip()
		ID, sentence = l.split("\t")
		r = sentence.split('(')[0]
		result.append(r)

with open("result.txt", 'w') as f:
		for d in result:
			f.write(d)
			f.write('\n')