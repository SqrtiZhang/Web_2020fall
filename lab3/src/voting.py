'''
Description: 
Author: Sqrti
Date: 2021-02-28 00:08:04
LastEditTime: 2021-02-28 03:13:05
LastEditors: Sqrti
'''
import numpy as np
svd_path = "./model/svd_float.txt"
knn_item_path = "./model/knn_item_float.txt"
knn_user_path = "./model/knn_user_float.txt"
knn_network_path = "./model/knn_network_float.txt"
soft_result_path = "./model/soft_voting_svd_item_user_network_3211.txt"

svd_around_path = "./model/svd_around.txt"
knn_item_around_path = "./model/knn_item_around.txt"
knn_user_around_path = "./model/knn_user_around.txt"

svd_result = np.loadtxt(svd_path)
knn_item_result = np.loadtxt(knn_item_path)
knn_user_result = np.loadtxt(knn_user_path)
knn_network_result = np.loadtxt(knn_network_path)

final_result = (3 * svd_result + 2 * knn_item_result + 1 * knn_user_result + 1 * knn_network_result) / 7
final_result = (np.around(final_result)).astype(np.int)

np.savetxt(soft_result_path, final_result, fmt="%d")
'''np.savetxt(svd_around_path, (np.around(svd_result)).astype(np.int), fmt="%d")
np.savetxt(knn_user_around_path, (np.around(knn_user_result)).astype(np.int), fmt="%d")
np.savetxt(knn_item_around_path, (np.around(knn_item_result)).astype(np.int), fmt="%d")'''