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

def reset_latent_space_image(df=None):
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

def medianAbsoluteDeviation(x, similarities):
    med = x[similarities].median()
    ad = abs(x[similarities] - med)
    exp_ad = np.exp(-ad.astype(np.float64))
    x[similarities] = exp_ad / exp_ad.sum()
    x[similarities] = ad / ad.sum()
    return x

def softmaxScheduler(x, similarities):
    x_t = torch.from_numpy(x[similarities].to_numpy().astype(np.float32))
    v_min, v_max = x_t.min(), x_t.max()
    new_min, new_max = -1, 1
    x_t = (x_t - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    x_t = torch.nn.functional.softmax(-x_t, dim=0) / (1/x_t.shape[0])
    x[similarities] = x_t.numpy()
    return x