import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import pickle

# Replace with the path to the folder containing the embeddings
database_embeddings_path = ['./embeddings/sampled_100k/csd_vit_large_sampled_100k_normal/1/database/embeddings_0.pkl'] 

# Open the file in binary read mode
def convert_embeddings(file_paths):
    database_embeddings = {'filenames': [], 'embeddings': []}
    for path in file_paths:
        with open(path, 'rb') as file:
            # Load the contents from the file
            curr_embeddings = pickle.load(file)
        database_embeddings['filenames'] += curr_embeddings['filenames']
        database_embeddings['embeddings'] += curr_embeddings['embeddings']
    
    return database_embeddings

database_embeddings = convert_embeddings(database_embeddings_path)
embeddings = {key: val for key,val in zip(database_embeddings['filenames'], database_embeddings['embeddings'])}

normalized_embeddings = normalize(np.array(list(embeddings.values())))
k = 10000
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(normalized_embeddings)
labels = kmeans.labels_

# Cluster centers
cluster_centers = kmeans.cluster_centers_

# Save the entire model
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)