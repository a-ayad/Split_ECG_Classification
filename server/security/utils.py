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
from scipy.special import kl_div, rel_entr
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
from contextlib import closing
from tqdm.notebook import tqdm
import math 

def get_unique_labels(df):	
    decimal = df.label.apply(lambda x: np.sum(x * 2**np.arange(x.size)[::-1]))
    #decimal = df.label
    ul = decimal.unique()
    un = decimal.value_counts().values
    ind = np.argsort(ul)
    ul = np.take_along_axis(ul, ind, axis=0)
    un = np.take_along_axis(un, ind, axis=0)
    return np.column_stack((ul, un))

# Converts a binary number to a decimal number
def binary_to_decimal(n):
    return np.sum(n * 2**np.arange(n.size)[::-1])	

# Gets a float as input and returns if it is a power of 2
def is_power_of_2(n):
    if not isinstance(n, (int, float)):
        n = np.sum(n * 2**np.arange(n.size)[::-1])
        
    n = int(n)	
    return (n & (n - 1) == 0) and n != 0

# Gets a binary number and returns if the ith bit is 1
def get_bit(n, i):	
    if not isinstance(n, (int, float)):
        n = np.sum(n * 2**np.arange(n.size)[::-1])
    n = int(n)
    return ((n & (1 << i)) >> i) == 1

def split_labels(df, split="min"):
    df = df.copy()
    n_bits = df.label.values[0].size
    
    if not split.startswith("knn"):
        df.label = df.label.apply(binary_to_decimal)
    
    if split == "min":
        df_split = pd.DataFrame()
        for i in range(0, n_bits):
            subset = df[df.label.apply(lambda x: get_bit(x, i))].copy()
            subset.label = 2**i
            df_split = pd.concat([df_split, subset], axis=0, ignore_index=True)
    elif split == "max":
        df_split = df[df.label.apply(is_power_of_2)]
    elif split == "dec":
        df_split = df.copy()
    elif split == "minmax":
        df_split = df[df.label.apply(is_power_of_2)]
        df_split_neg = df[df.label.apply(is_power_of_2)].copy()
        df_split_neg.label = 0
        df_split = pd.concat([df_split, df_split_neg], axis=0, ignore_index=True)
    elif split.startswith("knn"):
        df_split = knn_label_splitting(df, split)
        
    return df_split

def knn_label_splitting(df, split):
    params = split.split("_")
        
    df_X = df[df.label.apply(is_power_of_2)].copy()
    df_X.label = df_X.label.apply(binary_to_decimal)
    df_X0 = df[~df.label.apply(is_power_of_2)].copy()

    X = pool_latent_vectors(df_X.client_output.values.tolist(), pooling=None)
    X0 = pool_latent_vectors(df_X0.client_output.values.tolist(), pooling=None)
    neigh = KNeighborsClassifier(n_neighbors=int(params[1]), metric=params[2])
    neigh.fit(X, df_X.label)

    df_X0.label = 2 ** np.argmax(neigh.predict_proba(X0) * np.array(df_X0.label.to_list()), axis=1)
    
    return pd.concat([df_X, df_X0], axis=0, ignore_index=True)


def pool_latent_vectors(latent_vectors, pooling="average"):	
    latent_vectors = np.array(latent_vectors)
    if pooling == "max":
        latent_vectors = max_pool1d(torch.from_numpy(latent_vectors), kernel_size=latent_vectors.shape[-1]).squeeze(-1).numpy()
    elif pooling == "average":
        latent_vectors = avg_pool1d(torch.from_numpy(latent_vectors), kernel_size=latent_vectors.shape[-1]).squeeze(-1).numpy()
    else:
        latent_vectors = np.reshape(latent_vectors, (latent_vectors.shape[0], int(latent_vectors.shape[1]*latent_vectors.shape[2])))    
    return latent_vectors

def split_batch(batch):
    """Gets a pytorch tensor of shape (batch_size, ...). 
    Transforms the tensor to a numpy array and splits it into a list of 
    numpy arrays of shape (batch_size, 1, ...).
    """
    batch = batch.cpu().detach().numpy()	
    batch = np.split(batch, batch.shape[0], axis=0)	
    batch = [np.squeeze(t, axis=0) for t in batch]		
    return batch	

def reset_latent_space_image():
    df = pd.DataFrame(
        columns=[
            "client_output",
            "label",
            "step",
            "epoch",
            "stage",
            "loss",
            "client_id",
        ]
    )
    return df


def gaussian_kernel(d_ij, sigma=1):
    return np.exp(-(d_ij ** 2) / (2 * (sigma ** 2)))

def normalize(x, cols):
    x[cols] = x[cols] / np.sum(x[cols])
    return x

# def medianAbsoluteDeviation(x, similarities):
    
#     dataset = x[similarities].values

#     # Step 1: Calculate the median
#     median = np.median(dataset)

#     # Step 2: Compute the MAD
#     mad = np.median(np.abs(dataset - median))

#     # Step 3: Calculate the Manhatten distance for each data point to the median
#     manhatten_distances = np.abs(dataset - median)

#     # Step 4: Normalize the distances under a gaussian kernel to penalize outliers
#     normalized_contributions = gaussian_kernel(manhatten_distances, np.sqrt(mad/2))

#     x[similarities] = normalized_contributions
    
#     return x
    
# def medianAbsoluteDeviation(x, similarities, sigma=1/2):
#     med = x[similarities].median()
#     x[similarities] = abs(x[similarities] - med)
#     mad = x[similarities].median().values[0]
#     x[similarities] = x[similarities].applymap(lambda x: gaussian_kernel(x, np.sqrt(mad/2)))
#     return x

# def medianAbsoluteDeviation(x, similarities):
#     dataset = x[similarities].values
    
#     # Step 1: Calculate the median
#     median = np.median(dataset)

#     # Step 2: Compute the MAD
#     mad = np.median(np.abs(dataset - median))

#     # Step 3: Calculate the squared Mahalanobis distance for each data point
#     mahalanobis_distances = ((dataset - median) / mad) ** 2

#     # Step 4: Normalize the squared Mahalanobis distances
#     normalized_contributions = 1 - mahalanobis_distances / np.sum(mahalanobis_distances)

#     x[similarities] = normalized_contributions
    
#     return x

def medianAbsoluteDeviation(x, similarities):
    dataset = x[similarities].values
    
    # Step 1: Calculate the median
    median = np.median(dataset)

    # Step 2: Compute the MAD
    mad = np.median(np.abs(dataset - median))

    # Step 3: Calculate the modified Mahalanobis distance for each data point
    mahalanobis_distances = np.exp(-((dataset - median) ** 2) / mad)
    
    # Step 4: Normalize the modified Mahalanobis distances using the max distance
    normalized_contributions = mahalanobis_distances / np.max(mahalanobis_distances)

    x[similarities] = normalized_contributions
    
    return x

def softmaxScheduler(x, similarities):
    # x_t = torch.from_numpy(x[similarities].to_numpy().astype(np.float32))
    # v_min, v_max = x_t.min(), x_t.max()
    # new_min, new_max = -1, 1
    # x_t = (x_t - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    x_t = x[similarities]
    x_t = torch.nn.functional.softmax(x_t, dim=0) / (1/x_t.shape[0])
    x[similarities] = x_t.numpy()
    return x

def rolling_membership_diff(df_base, method="kernel", ref=None):
    df_plot = pd.DataFrame(columns=["epoch", "client_id", "re", "kl", "sre", "skl"])
    P = lambda X, c: X[X.client_id == c].sort_values("label")[method]
    
    old_p_k = None
    for epoch in df_base.epoch.sort_values().unique():
        df = df_base[df_base.epoch == epoch]
    
        if ref is not None:
            old_p_k = P(df, ref)
            
        for client_id in df_base.client_id.sort_values().unique():    
            p_k = P(df, client_id)
            
            if old_p_k is not None:
                p = old_p_k.values
                q = p_k.values
                            
                kl = kl_div(p, q).sum()
                re = rel_entr(p, q).sum()
                skl = kl + kl_div(q, p).sum()
                sre = re + rel_entr(q, p).sum()
                td = [re, kl, sre, skl]
                    
            else:
                td = [0., 0., 0., 0.]
                        
            df_plot.loc[len(df_plot)] = [epoch, client_id, *td]
            
            if ref is None:
                if old_p_k is None:
                    old_p_k = p_k
                elif re < 0.0:
                    old_p_k = p_k
                    
    return df_plot

def rolling_membership_diff(df_base, method="kernel", ref=None, div="skl"):
    df_plot = pd.DataFrame(columns=["epoch", "client_id", "re", "kl", "sre", "skl"])
    P = lambda X, c: X[X.client_id == c].sort_values("label")
    
    old_p_k = None
    for epoch in df_base.epoch.sort_values().unique():
        df = df_base[df_base.epoch == epoch]
    
        if ref is not None:
            old_p_k = P(df, ref)
        else:
            if len(df_plot) > 0:
                c = df_plot[df_plot.epoch == epoch - 1].reset_index()[div].argmin()
                old_p_k = P(df, c)
            else:
                old_p_k = P(df, 0)
            
        for client_id in df_base.client_id.sort_values().unique():    
            p_k = P(df, client_id)
            
            p_k = p_k[p_k.label.isin(old_p_k.label)]
            old_p_k = old_p_k[old_p_k.label.isin(p_k.label)]
        
            
            p = old_p_k[method].values
            q = p_k[method].values
                        
            kl = kl_div(p, q).sum()
            re = rel_entr(p, q).sum()
            skl = kl + kl_div(q, p).sum()
            sre = re + rel_entr(q, p).sum()
            td = [re, kl, sre, skl]
                        
            df_plot.loc[len(df_plot)] = [epoch, client_id, *td]
                    
    return df_plot


def exp_decay(row):
    x, t = row["cum_div"], row["epoch"]
    row["cum_div"] = 1 - np.exp(-x ** (t+1))
    return row