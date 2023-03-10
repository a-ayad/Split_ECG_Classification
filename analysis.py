import numpy as np
import torch 
import pickle
import pandas as pd
import scipy.spatial as sp
from functools import partial
import multiprocessing
#from tqdm import tqdm
import tqdm
from tqdm.contrib.concurrent import process_map


# Weights classes of a multi-label dataset based on the number of samples in each class. Every unique combination of labels is considered a class. Uses softmax to normalize the weights.
def class_weights(dataset):		
    # Get the number of samples in each class
    class_counts = np.zeros(dataset.num_classes)
    for i in range(dataset.num_classes):
        class_counts[i] = np.sum(dataset.labels[:, i])
    
    # Normalize the weights
    class_weights = np.exp(class_counts)
    class_weights = class_weights / np.sum(class_weights)
    
    return class_weights	

# Gets a pandas series of 5D numpy arrays. Filters out all arrays, which have a value of 0 in the 3rd dimension.
def filter_labels(df, idx, val=1):	
    return df[df.label.apply(lambda x: x[idx] == val)]

# Gets a numpy array as input. Returns a numpy array with all possible 2-combinations of the input. The order of the combinations is not important and combinations with the same elements are not included.
def get_similarities(X, similarity_functions):	
    similarities = {}	
    for s in similarity_functions:
        similarities[s] = sp.distance.pdist(X, s)
    return similarities

def get_unique_labels(df):	
    decimal = df.label.apply(lambda x: np.sum(x * 2**np.arange(x.size)[::-1]))
    ul = decimal.unique()
    un = decimal.value_counts().values
    ind = np.argsort(ul)
    ul = np.take_along_axis(ul, ind, axis=0)
    un = np.take_along_axis(un, ind, axis=0)
    return np.column_stack((ul, un))

def per_label_similarities(group, similarities):	
    latent_vectors = group[1].client_output_pooled.values
    latent_vectors = np.array(latent_vectors.tolist())
    sim = get_similarities(latent_vectors, similarities)
    sim["epoch"] = group[0][0]
    sim["label"] = group[0][1]
    return pd.DataFrame(sim)

def compute_in_parallel(similarities, groups):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    r = pool.map(partial(per_label_similarities, similarities=similarities), samples_d.groupby(["epoch", "label"]))
    return r

if __name__ == '__main__':
    image_path = "/home/mohkoh/Projects/Split_ECG_Classification/latent_space/single_client_honest/client_1.pickle"
    similarities = ["cosine", "euclidean", "cityblock", "correlation", "jaccard"]
    client1 = pickle.load(open(image_path, "rb"))
    samples = client1["samples"]
    unique_labels = get_unique_labels(samples)
    sim_list_per_epoch = pd.DataFrame(columns=["epoch", "label"] + similarities)
    samples_d = samples.copy()
    samples_d.label = samples_d.label.apply(lambda x: np.sum(x * 2**np.arange(x.size)[::-1]))
    groups = samples_d.groupby(["epoch", "label"])
    r = compute_in_parallel(similarities, groups)
    print("Done")