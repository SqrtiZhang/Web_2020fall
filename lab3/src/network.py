'''
Description: 
Author: Sqrti
Date: 2021-02-27 22:19:24
LastEditTime: 2021-02-27 22:54:21
LastEditors: Sqrti
'''
import pickle
import numpy as np

def user_follow(relation_path, save_path):
	network = {}
	with open(relation_path, 'r') as f:
		for line in f:
			item = line.split(',')
			user = item[0].split(':')[0]
			network[user] = item[1:-1]
			network[user].append(item[0].split(':')[1])
	pickle.dump(network, open(save_path, 'wb'))

def cac_sim(network_path, sim_path):
	network = pickle.load(open(network_path, 'rb'))
	users = network.keys()
	sim_mat = np.zeros((3000, 3000))
	for user in users:
		id = int(user)
		for other in users:
			other_id = int(other)
			if(sim_mat[other_id, id] > 0):
				sim_mat[id, other_id] = sim_mat[other_id, id]
			else:
				user_following = set(network[user])
				other_following = set(network[other])
				sim_mat[id, other_id] = len(user_following.intersection(other_following))*1.0/np.sqrt((len(user_following)+1) * (len(other_following)+1))
				#print(sim_mat[id, other_id])
				
	np.save(sim_path, sim_mat)

relation_path = './data/relation.txt'
save_path = './data/network.pkl'

#user_follow(relation_path, save_path)
sim_path = './data/sim_mat_network.npy'
cac_sim(save_path, sim_path)