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

# Plots the similarity measures over epochs for each client for a given label.
def plot_similarity(df, label, similarity):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", constrained_layout=True)
    fig.suptitle("Similarity: "+ similarity + ", Label: {}".format(label), size=16)
    df = df[df["label"] == label]
    for client_id, df_client in df.groupby("client_id"):
        df_client = df_client.sort_values("epoch")
        ax.plot(df_client["epoch"], df_client[similarity].apply(lambda x: x[0]), label="Client {}".format(client_id))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Similarity")
    ax.legend()
    plt.show()	
    
# Plots all similarity measures in one plot for a given label.
def plot_all_similarities(df_all, moment="mean", similarities=["euclidean", "cosine"]):	
    if moment == "mean":
        num_moment = 0
    elif moment == "std" or moment == "var":
        num_moment = 1
    elif moment == "median":
        num_moment = 2	
        
    labels = list(df_all["label"].unique())
    supfig = plt.figure(constrained_layout=True, figsize=(20, 10 * len(labels)))
    supfig.suptitle(f"Similarity Measures (per-class {moment})", fontsize='xx-large')
    subfigs = supfig.subfigures(len(labels), 1, facecolor="white", hspace=0.05)
    

    for fig_idx, subfig in enumerate(subfigs.flat):
        subfig.suptitle("Label: {}".format(labels[fig_idx]), fontsize='x-large')
        ax = subfig.subplots(2, 3)
        df = df_all[df_all["label"] == labels[fig_idx]]
        for idx, similarity in enumerate(similarities):
            for client_id, df_client in df.groupby("client_id"):
                df_client = df_client.sort_values("epoch")
                df_client[similarity] = df_client[similarity].apply(lambda x: x[num_moment])
                if moment == "std":
                    df_client[similarity] = df_client[similarity].apply(lambda x: np.sqrt(x))
                ax[idx // 3, idx % 3].plot(df_client["epoch"], df_client[similarity], label="Client {}".format(client_id))
            ax[idx // 3, idx % 3].set_title(similarity)
            ax[idx // 3, idx % 3].set_xlabel("Epoch")
            ax[idx // 3, idx % 3].set_ylabel("Similarity")
            ax[idx // 3, idx % 3].legend()
    plt.show()


def init_subplot(dim, nrows, ncols):
    if dim == 2:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 10 * nrows), facecolor="white", constrained_layout=True)
    elif dim == 3:
        fig, ax = plt.subplots(
            nrows=nrows, ncols=ncols,
        figsize=(10 * ncols, 10 * nrows),
        facecolor="white",
        constrained_layout=True,
        subplot_kw={"projection": "3d"},
    )
        
    return fig, ax    

def plot_2d(ax, points, points_color, title, label):
    x, y = points.T
    col = ax.scatter(x, y, c=points_color, s=50, alpha=0.8, label=label)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.legend()#loc='upper right', bbox_to_anchor=(0.5, 0.5))
    
    return col
    
def plot_3d(ax, points, points_color, title, label, view_init=(9, -60)):
    x, y, z = points.T
    
    ax.set_title(title, y=0.9, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=5, alpha=0.8, label=label)
    ax.view_init(*view_init)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.85))
    
    return col

def plot_embeddings(n_components, ax, points, points_color, title, label, **kwargs):
    if n_components == 2:
        plot_2d(ax, points, points_color, title, label, **kwargs)
    elif n_components == 3:
        plot_3d(ax, points, points_color, title, label, **kwargs)

def client_tsne(samples, params, pooling=None, pca=None):
    X = pool_latent_vectors(samples.client_output.to_list(), pooling=pooling)
    
    if pca:
        X = PCA(n_components=pca).fit_transform(X)
        
    X_embedded = TSNE(**params, n_jobs=5).fit_transform(X)
    label_list = np.array(samples.label.values.tolist()).astype(int)
    
    return X_embedded, label_list

def per_epoch_tsne(epochs, base_path, num_clients, params, pooling=None, pca=None, malicious_id=None, num_labels=5, split="min"):
    
    # Initialize subplots
    num_subplots = num_clients if malicious_id is None else 2
    num_epochs = len(epochs)
    fig, ax = init_subplot(params["n_components"], num_subplots, num_epochs)
    fig.suptitle("t-SNE Visualizations for epochs: {}".format(epochs), size=20)
    
    if split != "dec":
        cmap = plt.cm.get_cmap("viridis", int(2 ** (num_labels - 1)))
        norm = mpl.colors.BoundaryNorm((2**np.arange(num_labels)), cmap.N)
    else:
        cmap = plt.cm.get_cmap("viridis", int(2 ** (num_labels)))
        norm = mpl.colors.BoundaryNorm((np.arange(2 ** num_labels - 1)), cmap.N)
    
    # Get per epoch t-SNE embeddings for all clients and plot them
    for client_id in tqdm(range(1, num_subplots + 1), desc="Client", position=0):
        for idx, epoch in tqdm(enumerate(epochs), total=num_epochs, desc="Epoch", position=1, leave=False):
            samples = pd.read_pickle(os.path.join(base_path, "client_" + str(client_id), "epoch_" + str(epoch) + ".pickle"))
            samples = split_labels(samples, split=split)
            embeddings, labels = client_tsne(samples, params=params, pooling=pooling, pca=pca)
            label = "Client " + str(client_id)
            title = "Epoch " + str(epoch) if client_id == 1 else None
            plot_embeddings(params["n_components"], ax[client_id - 1, idx], embeddings, cmap(labels), title, label)
                
    fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax[-1, -1], orientation="horizontal", shrink=0.9, aspect=60, pad=0.01)            
    plt.show()
    return fig, ax


# Gets a dataframe with columns client_id, epoch, and loss. Plots the loss over epochs for each client.
def plot_loss(df, moment="mean"):	
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", constrained_layout=True)
    fig.suptitle("Loss over Epochs for Moment: {}".format(moment), size=16)
    for client_id, df_client in df.groupby("client_id"):
        df_client = df_client.sort_values("epoch")
        ax.plot(df_client["epoch"], df_client["loss"], label="Client {}".format(client_id))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()
    