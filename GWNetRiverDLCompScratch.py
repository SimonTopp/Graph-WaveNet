import pickle
import numpy as np
import pandas as pd

###### Distance Matrix Comparison

##GWNET
pickle_file = 'data/sensor_graph/adj_mx.pkl'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f, encoding='latin1')

###US
dm = np.load('../../gits/river-dl/DRB_data/distance_matrix_subset.npz')

def prep_adj_matrix(infile, dist_type, out_file=None):
    """
    process adj matrix.
    **The resulting matrix is sorted by seg_id_nat **
    :param infile:
    :param dist_type: [str] type of distance matrix ("upstream", "downstream" or
    "updown")
    :param out_file:
    :return: [numpy array] processed adjacency matrix
    """
    adj_matrices = np.load(infile)
    adj = adj_matrices[dist_type]
    adj = sort_dist_matrix(adj, adj_matrices["rowcolnames"])
    adj = np.where(np.isinf(adj), 0, adj)
    adj = -adj
    mean_adj = np.mean(adj[adj != 0])
    std_adj = np.std(adj[adj != 0])
    adj[adj != 0] = adj[adj != 0] - mean_adj
    adj[adj != 0] = adj[adj != 0] / std_adj
    adj[adj != 0] = 1 / (1 + np.exp(-adj[adj != 0]))

    I = np.eye(adj.shape[0])
    A_hat = adj.copy() + I
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)
    if out_file:
        np.savez_compressed(out_file, dist_matrix=A_hat)
    return A_hat

def sort_dist_matrix(mat, row_col_names):
    """
    sort the distance matrix by seg_id_nat
    :return:
    """
    df = pd.DataFrame(mat, columns=row_col_names, index=row_col_names)
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    return df


our_dm = prep_adj_matrix('../../gits/river-dl/DRB_data/distance_matrix_subset.npz', 'upstream')

import matplotlib.pyplot as plt

figus = plt.imshow(our_dm)
figthem = plt.imshow(pickle_data[-1])

########
##NOTE TO SELF, change line 42 in train to just adj.matrix
######

############ Input data comparison

####US
prepped = np.load('../../gits/river-dl/test_val_functionality/prepped.npz')


##### Them
gwn_train = np.load('data/METR-LA/train.npz')
gwn_train.shape