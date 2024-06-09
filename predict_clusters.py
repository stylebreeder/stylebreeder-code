import pickle
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import sys

def convert_embeddings(file_paths):
    database_embeddings = {'filenames': [], 'embeddings': []}
    for path in file_paths:
        with open(path, 'rb') as file:
            # Load the contents from the file
            curr_embeddings = pickle.load(file)
        database_embeddings['filenames'] += curr_embeddings['filenames']
        database_embeddings['embeddings'] += curr_embeddings['embeddings']
    
    return database_embeddings

# Load embeddings
folder_path = sys.argv[1] # Replace with the path to the folder containing the embeddings
# folder_path = './embeddings/csd_vit_large_artbreeder_normal/1/database/' # Replace with the path to the folder containing the embeddings
database_embeddings_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

database_embeddings = convert_embeddings(database_embeddings_path)
embeddings = {key: val for key,val in zip(database_embeddings['filenames'], database_embeddings['embeddings'])}

# Normalize embeddings
normalized_embeddings = normalize(np.array(list(embeddings.values())))

# Load the entire model
# kmeans_path = './kmeans_model.pkl'
kmeans_path = sys.argv[2]
with open(kmeans_path, 'rb') as file:
    loaded_kmeans = pickle.load(file)

# Predict using the loaded model
predicted_cluster_labels = loaded_kmeans.predict(normalized_embeddings)
save_path = 'predicted_cluster_labels.txt'
np.savetxt(save_path, predicted_cluster_labels, fmt='%d')
print("Predictions from loaded model:", predicted_cluster_labels)