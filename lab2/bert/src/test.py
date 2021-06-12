'''
Description: 
Author: Sqrti
Date: 2021-01-03 10:33:55
LastEditTime: 2021-01-03 11:11:19
LastEditors: Sqrti
'''
import numpy as np
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
#from transformers import BertPreTrainedModel
import random
from transformers import BertModel
from data_process import map_id_rel

rel2id, id2rel = map_id_rel()
USE_CUDA = torch.cuda.is_available()

def test(net_path,text_list):
	max_length=128
	net=torch.load(net_path)
	net.eval()
	if USE_CUDA:
		net = net.cuda()
	rel_list = []
	correct=0
	total=0
	with torch.no_grad():
		for text in text_list:
			sent = text
			tokenizer = BertTokenizer.from_pretrained(r'../../../data/Sqrti/bert-uncase/bert-base-uncased')
			indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
			avai_len = len(indexed_tokens)
			while len(indexed_tokens) < max_length:
				indexed_tokens.append(0)  # 0 is id for [PAD]
			indexed_tokens = indexed_tokens[: max_length]
			indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

			# Attention mask
			att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
			att_mask[0, :avai_len] = 1
			if USE_CUDA:
				indexed_tokens = indexed_tokens.cuda()
				att_mask = att_mask.cuda()

			if USE_CUDA:
				indexed_tokens=indexed_tokens.cuda()
				att_mask=att_mask.cuda()
			outputs = net(indexed_tokens, attention_mask=att_mask)
			# print(y)
			logits = outputs[0]
			_, predicted = torch.max(logits.data, 1)
			result=predicted.cpu().numpy().tolist()[0]
			#print("Source Text: ",text)
			#print(" Predict Relation: ",id2rel[result])
			
			#print('\n')
			rel_list.append(id2rel[result])
	#print(correct," ",total," ",correct/total)
	return rel_list

	


def caculate_acc():
	text_list=[]
	with open(r"../data/submit_test.json", 'r', encoding='utf-8') as load_f:
		lines = load_f.readlines()
		for line in lines:
			dic = json.loads(line)
			text_list.append(dic['text'])

	if len(text_list) == 0:
		print("No sample: ")
	else:
		result = test(r'../model/bert.pth', text_list)
		savepath = r'../output/output.txt'
		with open(savepath, 'w') as f:
			for r in result:
				f.write(r)
				f.write('\n')

	
caculate_acc()
