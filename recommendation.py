import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import pickle
import sys
import os
from surprise import Dataset, NormalPredictor, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import SVD, accuracy

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

def process_path(folder_path):
    database_embeddings_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    database_embeddings = convert_embeddings(database_embeddings_path)
    embeddings = {key: val for key,val in zip(database_embeddings['filenames'], database_embeddings['embeddings'])}
    return database_embeddings, embeddings

if __name__ == '__main__':

    folder_path = sys.argv[1] # path to the folder containing the embeddings

    database_embeddings_test, test_embeddings = process_path(folder_path)
    normalized_embeddings = normalize(np.array(list(test_embeddings.values())))

    # Load the entire model
    kmeans_path = sys.argv[2]
    with open(kmeans_path, 'rb') as file:
        loaded_kmeans = pickle.load(file)

    # Predict using the loaded model
    predicted_labels = loaded_kmeans.predict(normalized_embeddings)

    database_embeddings_test['labels'] = predicted_labels

    user_cluster = pd.DataFrame(database_embeddings_test)
    user_cluster['image_key'] = user_cluster['filenames'].str.replace('.jpeg', '', regex=False)

    source_path = 'artbreeder_collage_data.csv'
    source_df = pd.read_csv(source_path)
    user_cluster['creator_id'] = source_df[source_df['image_key'].isin(user_cluster['image_key'])]['creator_id'].reset_index(drop=True)

    # Count the number of elements in each cluster
    cluster_counts = np.bincount(loaded_kmeans.labels_)

    # Get clusters with more than 10 elements
    selected_clusters = np.where(cluster_counts > 10)[0]

    # Group by creator and labels and count the number of elements in each group
    k = 10000
    label_counts = user_cluster.groupby(['creator_id', 'labels']).size().unstack(fill_value=0)
    label_counts = label_counts.reindex(columns=np.arange(k), fill_value=0)

    # Vectorize the number of images per cluster for each creator and normalize
    creator_vectors = {creator: row.values / np.linalg.norm(row.values) for creator, row in label_counts.iterrows()}
    filtered_creator_vectors = {
        creator: vector[selected_clusters] / np.linalg.norm(vector[selected_clusters])
        for creator, vector in creator_vectors.items()
        if np.linalg.norm(vector[selected_clusters]) > 0
    }
    triples = [(creator_id, selected_clusters[i], val) for creator_id, vector in filtered_creator_vectors.items()
            for i, val in enumerate(vector) if val > 0]

    # Convert to DataFrame and load into a surprise dataset
    recommend_df = pd.DataFrame(triples, columns=['userID', 'itemID', 'rating'])
    reader = Reader(rating_scale=(0, 1))

    # Split the dataset into training and test sets
    recommend_data = Dataset.load_from_df(recommend_df[["userID", "itemID", "rating"]], reader)
    trainset, testset = train_test_split(recommend_data, test_size=0.2, random_state=0)

    # Cross validate using SVD
    algo = SVD(random_state=0)
    print(cross_validate(algo, recommend_data, measures=["RMSE", "MAE"], cv=5, verbose=True))