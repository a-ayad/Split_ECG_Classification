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

def simple_membership(nX, y):
    nY = y[nX]
    comp = nY[:, 0]
    p_k = (nY == comp[:, None]).sum(axis=1) / 10 - 1/10
    # p_k = {k: p_k[y == k].mean() for k in np.unique(y)}
    p_k = [p_k[y == k].mean() for k in np.unique(y)]
    p_k = p_k / np.sum(p_k)
    p_k = {k: p_k[i] for i, k in enumerate(np.unique(y))}
    return p_k

def kernel_membership(nX, dX, y, sigma=1.0):
    dY = np.exp(-(dX)[:, 1:] ** 2 / sigma ** 2)
    nY = y[nX]
    comp = nY[:, 0]
    nY = nY[:, 1:]
    p_k = (dY * (nY == comp[:, None])).sum(axis=1) / dY.sum(axis=1)
    np.nan_to_num(p_k, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # p_k = {k: p_k[y == k].mean() for k in np.unique(y)}
    p_k = [p_k[y == k].mean() for k in np.unique(y)]
    p_k = p_k / np.sum(p_k)
    p_k = {k: p_k[i] for i, k in enumerate(np.unique(y))}
    return p_k

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
    
    if multi_idx is None:
        idxs = range(0, X.shape[0])
        multi_idx =  pd.MultiIndex.from_tuples(list(itertools.combinations(idxs, 2)), names=["i", "j"])
    
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

def per_class_membership(X_c, knn_params, sigma=1.0, pca_params=None):    
    df_scores = pd.DataFrame(columns=["label", "simple", "kernel"])
    # Get high dimensional vectors
    latent_vectors = np.array(X_c["client_output"].to_list())
    
    # Get lower dimensional vectors if pca is used
    if pca_params:
        pca = PCA(**pca_params).fit(latent_vectors)
        X = pca.transform(latent_vectors)
    else:
        X = latent_vectors
        
    y = X_c.label.to_numpy()

    neigh = KNeighborsClassifier(**knn_params)
    neigh.fit(X, y)
    dX, nX = neigh.kneighbors(X, return_distance=True)
    simple, kernel = simple_membership(nX, y), kernel_membership(nX, dX, y, sigma=sigma)
    
    for k in simple.keys():
        df_scores.loc[len(df_scores)] = [k, simple[k], kernel[k]]

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

def per_epoch_scores(df_group, method="density", pooling="average", split="min", **kwargs):
    client_id, epoch = df_group[0]
    df = df_group[1]

    X_c = split_labels(df, split)
    X_c.client_output = pool_latent_vectors(X_c.client_output.to_list(), pooling=pooling).tolist()
    X_c.reset_index(drop=True, inplace=True)

    if method == "density":
        df_scores = per_class_densities(X_c, **kwargs)
    elif method == "similarity":
        df_scores = per_class_similarities(X_c, **kwargs)
    elif method == "membership":
        df_scores = per_class_membership(X_c, **kwargs)

    df_scores["epoch"] = epoch
    df_scores["client_id"] = client_id
    
    return df_scores    

def per_client_scores(client_id, epochs, **kwargs):
    df_scores = pd.DataFrame()
    
    if isinstance(epochs, int):
        epochs_iter = range(1, epochs + 1)
    elif isinstance(epochs, list):
        epochs_iter = epochs
    
    with tqdm(total=epochs, desc=f"Client {client_id}", position=client_id, leave=False) as pbar:
        for epoch in epochs_iter:
            df_scores = pd.concat([df_scores, per_epoch_scores(client_id, epoch, **kwargs)], ignore_index=True)
            pbar.update(1)
    df_scores["client_id"] = client_id
    return df_scores

def client_scores(df_base, num_clients, epochs, num_workers=5, **kwargs):
    # Load data
    if isinstance(epochs, int):
        epochs_iter = range(1, epochs + 1)
    elif isinstance(epochs, list):
        epochs_iter = epochs
        
    if isinstance(df_base, pd.DataFrame):
        df = df_base[(df_base.client_id < num_clients) & (df_base.epoch.isin(epochs_iter))]
    elif isinstance(df_base, str):
        df = pd.DataFrame()
        for idx in tqdm(range(num_clients), desc=f"Load Client Data Frames"):
            for epoch in epochs_iter:
                client_path = os.path.join(df_base, "client_" + str(idx))
                df_epoch = pd.read_pickle(os.path.join(client_path, "epoch_{}.pickle".format(epoch)))
                df_epoch["client_id"] = idx
                df = pd.concat([df, df_epoch], axis=0, ignore_index=True)
    
    df_scores = pd.DataFrame()
    df_groups = df.groupby(["client_id", "epoch"])
    
    partial_per_epoch_scores = partial(per_epoch_scores, **kwargs)
    
    # Compute in parallel
    with closing(multiprocessing.Pool(processes=num_workers)) as p:
        with tqdm(total=df_groups.ngroups, desc=f"|Client x Epoch|") as pbar:
            for r in p.imap_unordered(partial_per_epoch_scores, df_groups):
                df_scores = pd.concat([df_scores, r], ignore_index=True, copy=False)
                pbar.update()
    p.join()
    
    return df_scores

def loss_contributions(df_base, num_clients, epochs=30, moment="mean"):
    df_clients_loss = pd.DataFrame(columns=["client_id", "epoch", "loss"])
    
    if isinstance(epochs, int):
        epochs_iter = range(1, epochs + 1)
    elif isinstance(epochs, list):
        epochs_iter = epochs
        
    if isinstance(df_base, pd.DataFrame):
        df = df_base[(df_base.client_id < num_clients) & (df_base.epoch.isin(epochs_iter))]
    elif isinstance(df_base, str):
        df = pd.DataFrame()
        for idx in tqdm(range(num_clients), desc=f"Load Client Data Frames"):
            for epoch in epochs_iter:
                client_path = os.path.join(df_base, "client_" + str(idx))
                df_epoch = pd.read_pickle(os.path.join(client_path, "epoch_{}.pickle".format(epoch)))
                df_epoch["client_id"] = idx
                df = pd.concat([df, df_epoch], axis=0, ignore_index=True)
        
    df_clients_loss = df_base.groupby(["client_id", "epoch"])
    
    if moment == "mean":
        df_clients_loss = df_clients_loss.loss.mean()
    elif moment == "std" or moment == "var":
        df_clients_loss = df_clients_loss.loss.var()
        if moment == "std":
            df_clients_loss = np.sqrt(df_clients_loss)
    elif moment == "sum":
        df_clients_loss = df_clients_loss.loss.sum().astype(np.float32)
    elif moment == "median":
        df_clients_loss = df_clients_loss.loss.median()
        
    return df_clients_loss