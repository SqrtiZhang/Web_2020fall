'''
Description: 
Author: Sqrti
Date: 2021-02-27 02:26:37
LastEditTime: 2021-02-27 22:08:10
LastEditors: Sqrti
'''
# -*- coding: utf-8 -*-
import pickle
train_data_path = "./data/training.dat"
test_data_path = "./data/testing.dat"
simple_train_data_path =  "./data/train_simple.dat"
simple_test_data_path =  "./data/test_simple.dat"

def get_simple(origin_data_path, obj_data_path, is_test=False):
	with open(obj_data_path, 'w', encoding='utf-8') as fw:
		with open(origin_data_path, 'r', encoding='utf-8') as fr:
			if not is_test:
				for line in fr:
					fw.write(','.join(line.split(',')[0:4]) + '\n')
			else:
				for line in fr:
					fw.write(','.join(line.split(',')[0:3]) + '\n')

def movieid2id(data_path, save_path, new_data_path, is_test=False):
	id = 0
	movie_id={}
	with open(new_data_path, 'w', encoding='utf-8') as fw:
		with open(data_path, 'r', encoding='utf-8') as fr:
			for line in fr:
				item =  line.split(',')
				movieid = item[1]
				if movieid not in movie_id:
					movie_id[movieid] = id
					id += 1
				if not is_test:
					fw.write(','.join((item[0], str(movie_id[movieid]), item[2], item[3], item[4])))
	
	pickle.dump(movie_id, open(save_path, "wb"))

def getnewmovieid(data_path, new_data_path, map_path,is_test=False):
	movie_id = pickle.load(open(map_path, "rb"))
	with open(new_data_path, "w", encoding='utf-8') as fw:
		with open(data_path, 'r', encoding='utf-8') as fr:
			for line in fr:
				item =  line.split(',')
				movieid = item[1]
				if movieid in movie_id:
					movieid = movie_id[movieid]
				else:
					movieid = -1
				if not is_test:
					fw.write(','.join((item[0], str(movieid), item[2], item[3], item[4])))
				else:
					fw.write(','.join((item[0], str(movieid), item[2]))+'\n')
					
save_path =  "./data/movieid2id.pkl"
new_train_data_path =  "./data/train_map.dat"
new_test_data_path =  "./data/test_map.dat"
new_simple_test_data_path = "./data/test_simple_map.dat"
#movieid2id(train_data_path, save_path, new_train_data_path)
#get_simple(new_train_data_path, simple_train_data_path)
getnewmovieid(test_data_path, new_test_data_path, save_path, True)
#get_simple(new_test_data_path, new_simple_test_data_path, True)
#get_simple(test_data_path, simple_test_data_path, True)