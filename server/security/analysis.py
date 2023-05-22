import itertools
import time
import numpy as np
import os
from torch.nn.functional import max_pool1d, avg_pool1d
import torch
import pickle
import pandas as pd
import scipy.spatial as sp
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
from contextlib import closing
from tqdm.notebook import tqdm_notebook as tqdm
import math  
from .utils import *

def gaussian_kernel(d_ij, sigma=1):
    return np.exp(-d_ij ** 2 / (2 * (sigma ** 2)))

def get_p_ij(X, sigma=1, s="euclidean", Z=None, multi_idx=None):
    d_ij = gaussian_kernel(sp.distance.pdist(X, s), sigma)
    
    if Z is None:
        Z = np.sum(d_ij)
    
    m = X.shape[0]        
    
    if multi_idx is None:
        idxs = range(0, m)
        multi_idx =  pd.MultiIndex.from_tuples(list(itertools.product(idxs, idxs)), names=["i", "j"])
    
    P = d_ij / Z 
    
    P = pd.Series(multi_idx.map(lambda x: P[m * x[0] + x[1] - ((x[0] + 2) * (x[0] + 1)) // 2]), index=multi_idx)
            
    return P

# Gets a numpy array as input. Returns a numpy array with all possible 2-combinations of the input. The order of the combinations is not important and combinations with the same elements are not included.
def get_similarities(X, similarity_functions):	
    similarities = {}
    for s in similarity_functions:
        if X.shape[0] == 1:
            similarities[s] = np.array([0])
        else:
            similarities[s] = sp.distance.pdist(X, s)
    return similarities

def per_class_densities(X_c, similarities=["seuclidean"], sigma=1, multi_idx=None):
    df_scores = pd.DataFrame(columns=["label"] + similarities)
    X = np.array(X_c["client_output"].to_list())
    P = {s: get_p_ij(X, sigma=sigma, s=s, multi_idx=multi_idx) for s in similarities}
    
    X_cY = X_c.groupby(["label"])
    for y, X_cy in X_cY:
        Q_y = {"label": y}
        for s in similarities:
            P_s = P[s]
            alpha_y = 1 #(len(X_cy) / len(X_c))
            p_y = P_s[(P_s.index.get_level_values(0).isin(X_cy.index)) & (P_s.index.get_level_values(1).isin(X_cy.index) )].sum()
            z_y = P_s[(P_s.index.get_level_values(0).isin(X_cy.index)) | (P_s.index.get_level_values(1).isin(X_cy.index) )].sum()
            Q_y[s] = (p_y / z_y) * alpha_y
        df_scores.loc[len(df_scores)] = Q_y
    return df_scores

def per_class_similarities(X_c, similarities=["cosine"], aggregate="mean"):
    df_scores = pd.DataFrame(columns=["label"] + similarities)
    X_c = X_c.groupby(["label"])
    for label, group in X_c:
        latent_vectors = np.array(group["client_output"].to_list())
        sim = get_similarities(latent_vectors, similarities)
        sim["label"] = label
        df_scores.loc[len(df_scores)] = sim
        
    if aggregate == "mean":
        agg = np.mean
    elif aggregate == "median":
        agg = np.median
    elif aggregate == "var":
        agg = np.var
        
    for s in similarities:
        df_scores[s] = df_scores[s].apply(agg)
    return df_scores

def per_epoch_scores(epoch, client_id, base_path=None, df=None, method="density", pooling="average", split="min", **kwargs):
    if base_path is not None:
        df = pd.read_pickle(os.path.join(base_path, f"client_{client_id}", f"epoch_{epoch}.pickle"))
    X_c = split_labels(df, split)
    X_c.client_output = pool_latent_vectors(X_c.client_output.to_list(), pooling=pooling).tolist()
    X_c.reset_index(drop=True, inplace=True)

    if method == "density":
        df_scores = per_class_densities(X_c, **kwargs)
    elif method == "similarity":
        df_scores = per_class_similarities(X_c, **kwargs)

    df_scores["epoch"] = epoch
    df_scores["client_id"] = client_id
    
    return df_scores    

def per_client_scores(client_id, epochs, **kwargs):
    df_scores = pd.DataFrame()
    with tqdm(total=epochs, desc=f"Client {client_id}", position=client_id, leave=False) as pbar:
        for epoch in range(1, epochs + 1):
            df_scores = pd.concat([df_scores, per_epoch_scores(client_id, epoch, **kwargs)], ignore_index=True)
            pbar.update(1)
    df_scores["client_id"] = client_id
    return df_scores

def client_scores(num_clients, epochs, num_workers=5, **kwargs):
    df_scores = pd.DataFrame()
    
    for client_id in range(1, num_clients + 1):
        partial_per_epoch_scores = partial(per_epoch_scores, client_id=client_id, **kwargs)
    # Compute in parallel
        with closing(multiprocessing.Pool(processes=num_workers)) as p:
            with tqdm(total=epochs, desc=f"Client {client_id}") as pbar:
                for r in p.imap_unordered(partial_per_epoch_scores, np.arange(1, epochs+1)):
                    df_scores = pd.concat([df_scores, r], ignore_index=True, copy=False)
                    pbar.update()
    p.join()
    
    return df_scores

def loss_contributions(base_path, metadata, epochs=30, moment="mean"):
    df_clients_loss = pd.DataFrame(columns=["client_id", "epoch", "loss"])
    for idx in tqdm(range(1, metadata["num_clients"] + 1)):
        client_path = os.path.join(base_path, "client_" + str(idx))
        for epoch in range(1, epochs + 1):
            df = pd.read_pickle(os.path.join(client_path, "epoch_{}.pickle".format(epoch)))
            if moment == "mean":
                loss_moment = df.loss.mean()
            elif moment == "std" or moment == "var":
                loss_moment = df.loss.var()
                if moment == "std":
                    loss_moment = np.sqrt(loss_moment)
            elif moment == "sum":
                loss_moment = df.loss.sum()	
            elif moment == "median":
                loss_moment = df.loss.median()
            df_clients_loss.loc[len(df_clients_loss), df_clients_loss.columns] = [idx, epoch, loss_moment]
    return df_clients_loss
