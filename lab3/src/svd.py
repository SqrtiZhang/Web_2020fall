'''
Description: 
Author: Sqrti
Date: 2021-02-27 01:56:20
LastEditTime: 2021-02-28 02:57:52
LastEditors: Sqrti
'''
import surprise
from surprise import SVD
from surprise import Dataset, Reader, dump
from surprise.model_selection import cross_validate

import numpy as np
from six import iteritems


def train(train_data_path, save_path):
	reader = Reader(line_format='user item rating timestamp', sep=',')
	train_data = Dataset.load_from_file(train_data_path, reader=reader)
	
	#model = SVD()
	#model = surprise.KNNWithMeans()
	model = surprise.KNNWithMeans(sim_options={'user_based':False})
	cross_validate(model, train_data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
	dump.dump(save_path, algo=model)


def predict(test_data_path, model_path):
	model = dump.load(model_path)[1]
	with open("submission.txt", "w", encoding="utf-8") as fw:
		with open(test_data_path, "r", encoding="utf-8") as fr:
			for line in fr:
				item = line.split(',')
				fw.write(str((model.predict(item[0], item[1], item[2]).est)) + '\n')

def test(model_path, network_path, test_data_path):
	model = dump.load(model_path)[1]
	network_sim = np.load(network_path)

	usersim = model.sim
	inner_net = np.zeros(usersim.shape)

	raw2inner = model.trainset._raw2inner_id_users
	inner2raw = {inner: raw for (raw, inner) in iteritems(raw2inner)}

	for i in range(usersim.shape[0]):
		for j in range(usersim.shape[1]):
			rawid_i = int(inner2raw[i])
			rawid_j = int(inner2raw[j])
			inner_net[i, j] = network_sim[rawid_i, rawid_j]


	model.sim = (inner_net * 10 + usersim * 0) / 10

	with open("submission.txt", "w", encoding="utf-8") as fw:
		with open(test_data_path, "r", encoding="utf-8") as fr:
			for line in fr:
				item = line.split(',')
				fw.write(str(((model.predict(item[0], item[1], item[2]).est))) + '\n')




train_data_path = "./data/train_simple.dat"
save_path = "./model/knnmeans.model"
test_data_path = "./data/test_simple.dat"
network_path = "./data/sim_mat_network.npy"
test(save_path, network_path, test_data_path)
#train(train_data_path, save_path)
#predict(test_data_path, save_path)

"""
svd
Evaluating RMSE, MAE of algorithm SVD on 3 split(s).

                  Fold 1  Fold 2  Fold 3  Mean    Std
RMSE (testset)    1.3083  1.3095  1.3069  1.3082  0.0011
MAE (testset)     1.0156  1.0173  1.0151  1.0160  0.0009
Fit time          87.77   100.41  107.13  98.44   8.02
Test time         8.95    8.70    8.84    8.83    0.10\

"""

"""
Userbased knn
Evaluating RMSE, MAE of algorithm KNNWithMeans on 3 split(s).

                  Fold 1  Fold 2  Fold 3  Mean    Std
RMSE (testset)    1.3026  1.3040  1.3027  1.3031  0.0006
MAE (testset)     1.0225  1.0239  1.0235  1.0233  0.0006
Fit time          32.19   30.67   30.37   31.08   0.80
Test time         265.79  244.43  232.83  247.68  13.65

"""

"""
itembased knn 30
Evaluating RMSE, MAE of algorithm KNNWithMeans on 3 split(s).

                  Fold 1  Fold 2  Fold 3  Mean    Std     
RMSE (testset)    1.3938  1.3937  1.3917  1.3931  0.0010  
MAE (testset)     1.1155  1.1158  1.1156  1.1156  0.0001  
Fit time          132.26  127.26  128.35  129.29  2.15    
Test time         422.78  432.45  430.88  428.70  4.24   
"""

"""
itembased knn 40
Evaluating RMSE, MAE of algorithm KNNWithMeans on 3 split(s).

                  Fold 1  Fold 2  Fold 3  Mean    Std     
RMSE (testset)    1.3821  1.3843  1.3851  1.3838  0.0013  
MAE (testset)     1.1073  1.1088  1.1090  1.1084  0.0007  
Fit time          224.57  235.39  215.72  225.22  8.04    
Test time         452.49  468.17  455.30  458.65  6.83   

"""