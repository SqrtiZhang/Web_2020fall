import json
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import re

def setup_seed(seed):
	 torch.manual_seed(seed)
	 torch.cuda.manual_seed_all(seed)
	 np.random.seed(seed)
	 random.seed(seed)

def prepare_data(data_path):
	train_data_path = data_path + "train.txt"
	pattern_text = re.compile(r"\".*\"")
	pattern_rel = re.compile(r".*\(")
	pattern_ent1 = re.compile(r"\(.*\,")
	pattern_ent1 = re.compile(r"\,.*\)")
	info = []
	with open(train_data_path, "r") as load_f:
		info = []
		single_data = {}
		i = 1
		for line in load_f.readlines():
			if i % 2 ==1:
				single_data={}
				single_data['text'] = pattern_text.findall(line)[0][1:-1]
			else:
				single_data['rel'] = pattern_rel.findall(line)[0][:-1]
				single_data['ent1'] = pattern_ent1.findall(line)[0][1:-1]
				single_data['ent2'] = pattern_ent1.findall(line)[0][1:-1]
			i += 1
			info.append(single_data)
	
	train, other = train_test_split(info,test_size = 0.3)
	test, dev = train_test_split(other ,test_size = 0.2)

	train_json_path = data_path + "train.json"
	dev_json_path = data_path + "dev.json"
	test_json_path = data_path + "test.json"

	with open(train_json_path, "w") as dump_f:
		for i in train:
			a = json.dumps(i, ensure_ascii=False)
			dump_f.write(a)
			dump_f.write("\n")
	
	with open(dev_json_path, "w") as dump_f:
		for i in dev:
			a = json.dumps(i, ensure_ascii=False)
			dump_f.write(a)
			dump_f.write("\n")

	with open(test_json_path, "w") as dump_f:
		for i in test:
			a = json.dumps(i, ensure_ascii=False)
			dump_f.write(a)
			dump_f.write("\n")

def prepare_submit_data(data_path):
	test_data_path = data_path + "test.txt"
	pattern_text = re.compile(r"\".*\"")
	test = []
	with open(test_data_path, "r") as load_f:
		info = []
		single_data = {}
		for line in load_f.readlines():
			single_data={}
			single_data['text'] = pattern_text.findall(line)[0][1:-1]
			info.append(single_data)
		test = info
	test_json_path = data_path + "submit_test.json"

	with open(test_json_path, "w") as dump_f:
		for i in test:
			a = json.dumps(i, ensure_ascii=False)
			dump_f.write(a)
			dump_f.write("\n")

def map_id_rel():
	rel = ["Other", "Cause-Effect", "Component-Whole", "Entity-Destination", "Product-Producer", 
	"Entity-Origin", "Member-Collection", "Message-Topic", "Content-Container", 
	"Instrument-Agency"]

	id2rel = {}
	rel2id = {}

	for r in rel:
		rel2id[r] = len(rel2id)
	
	for r, i in rel2id.items():
		id2rel[i] = r

	return rel2id,id2rel

def load_train(data_path):
	rel2id,id2rel = map_id_rel()
	max_length=128
	tokenizer = BertTokenizer.from_pretrained(r'../../../data/Sqrti/bert-uncase/bert-base-uncased')
	train_data = {}
	train_data['label'] = []
	train_data['mask'] = []
	train_data['text'] = []

	with open(data_path, 'r') as load_f:        
		for line in load_f.readlines():
			dic = json.loads(line)
			if dic['rel'] not in rel2id:
				train_data['label'].append(0)
			else:
				train_data['label'].append(rel2id[dic['rel']])
			#sent=dic['ent1']+dic['ent2']+dic['text']
			sent = dic['text']
			indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
			avai_len = len(indexed_tokens)
			while len(indexed_tokens) <  max_length:
				indexed_tokens.append(0)  # 0 is id for [PAD]
			indexed_tokens = indexed_tokens[: max_length]
			indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

			# Attention mask
			att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
			att_mask[0, :avai_len] = 1
			train_data['text'].append(indexed_tokens)
			train_data['mask'].append(att_mask)
	return train_data

def deal_distr(data_path):

	rel2id, id2rel = map_id_rel()
	label = np.zeros(len(rel2id))
	with open(data_path, 'r') as load_f:
		for line in load_f.readlines():
			dic = json.loads(line)
			label[rel2id[dic['rel']]] += 1
	print("Distribution: ", label)



if __name__ == "__main__":
	#map_id_rel()
	data_path = r"../data/"
	deal_distr(data_path + 'train.json')
	#prepare_data(data_path)
	#prepare_submit_data(data_path)