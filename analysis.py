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

# Gets a numpy array as input. Returns a numpy array with all possible 2-combinations of the input. The order of the combinations is not important and combinations with the same elements are not included.
def get_similarities(X, similarity_functions):	
    similarities = {}
    for s in similarity_functions:
        if X.shape[0] == 1:
            similarities[s] = np.array([0])
        else:
            similarities[s] = sp.distance.pdist(X, s)
    return similarities

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
    
def per_label_similarities(epoch, base_path, similarities, aggregate=True, pooling="average", split="min"):	
    epoch_path = os.path.join(base_path, "epoch_" + str(epoch) + ".pickle")
    samples = pd.read_pickle(epoch_path)
    samples = split_labels(samples, split)
    samples = samples.groupby(["label"])
    df_epoch = pd.DataFrame(columns=["epoch", "label"] + similarities)
    for label, group in samples:
        latent_vectors = pool_latent_vectors(group.client_output.values.tolist(), pooling=pooling)
        sim = get_similarities(latent_vectors, similarities)
        sim["epoch"] = epoch
        sim["label"] = label
        df_epoch = pd.concat([df_epoch, pd.DataFrame([sim])], ignore_index=True)
    if aggregate:
        for s in similarities:
            df_epoch[s] = df_epoch[s].apply(lambda x: (np.mean(x), np.var(x), np.median(x)))
    return df_epoch

def pool_latent_vectors(latent_vectors, pooling="average"):	
    latent_vectors = np.array(latent_vectors)
    if pooling == "max":
        latent_vectors = max_pool1d(torch.from_numpy(latent_vectors), kernel_size=latent_vectors.shape[-1]).squeeze(-1).numpy()
    elif pooling == "average":
        latent_vectors = avg_pool1d(torch.from_numpy(latent_vectors), kernel_size=latent_vectors.shape[-1]).squeeze(-1).numpy()
    else:
        latent_vectors = np.reshape(latent_vectors, (latent_vectors.shape[0], int(latent_vectors.shape[1]*latent_vectors.shape[2])))    
    return latent_vectors

def compute_in_parallel(base_path, epochs, similarities, num_workers=5, save_path=None, aggregate=True, pooling="average", split="min"):	
    df = pd.DataFrame(columns=["epoch", "label"] + similarities)
    partial_per_label_similarities = partial(per_label_similarities, base_path=base_path, similarities=similarities, aggregate=aggregate, pooling=pooling, split=split)
    with closing(multiprocessing.Pool(processes=num_workers)) as p:
        with tqdm(total=epochs) as pbar:
            for r in p.imap_unordered(partial_per_label_similarities, np.arange(1, epochs+1)):
                df = pd.concat([df, r], ignore_index=True, copy=False)
                pbar.update()
    p.join()
    
    if save_path:
        df.to_pickle(os.path.join(base_path, save_path))
    return df

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
    