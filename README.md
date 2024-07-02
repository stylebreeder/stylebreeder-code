# Stylebreeder Clustering and Recommendation

First, download the model checkpoint for ViT-L [here](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view?usp=sharing)

Use `sample_imgs_and_to_csv.py` to create a csv file for the image dataset:

```
python sample_imgs_and_to_csv.py <IMAGES DIRECTORY PATH>
```

Then generate embeddings for the images:

```
python main_sim.py --dataset artbreeder -a vit_large --pt_style csd --feattype normal --world-size 1 --dist-url tcp://localhost:6001 -b 128 -j 8 --embed_dir ./embeddings --data-dir <IMAGE DIRECTORY PATH> --model_path <PATH TO VIT CHECKPOINT>
```

Then, either use the kmeans model to assign cluster labels or fit a kmeans model using `clustering.py` which will save the model in a `.pkl` file. The embedding directory should be something like `./embeddings/csd_vit_large_artbreeder_normal/1/database/`:

```
python predict_clusters.py <EMBEDDING DIRECTORY> kmeans_model.pkl
```

This will save the labels into a txt file titled `predicted_cluster_labels.txt`

To run recommendation, use the `recommendation.py` file and replace the path with the path to the embeddings you want to test.

```
python recommendation.py <TEST EMBEDDING DIRECTORY> kmeans_model.pkl
```
