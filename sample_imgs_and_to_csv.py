import os
import pandas as pd
import numpy as np
from PIL import Image
import sys

# Define the directory containing the JPG files
# directory_name = '<IMAGE_DIRECTORY>' # replace with name of image directory
directory_name = os.path.dirname(sys.argv[1]) # replace with name of image directory
# directory = f'./{directory_name}'
directory = os.path.join(os.getcwd(),directory_name)

# Function to check if an image is corrupted
def is_corrupted(filepath):
    try:
        img = Image.open(filepath)  # Open the image file
        img.verify()  # Verify that it is, indeed, an image
        return False
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image: {filepath}")
        return True

full_directory = os.listdir(directory)
sample_size = len(full_directory) # all images
sampled_files = np.random.choice(full_directory, size=sample_size, replace=False)
print(sampled_files)
print(sample_size)

# List all JPG files in the directory
files = [os.path.join(directory, file) for file in sampled_files 
         if file.endswith('.jpeg') and not is_corrupted(os.path.join(directory, file))]

print(files)

# Create a DataFrame with file paths
df = pd.DataFrame(files, columns=['path'])

print(df)

# Assign 'split' column
query_prop = 0.2
query_size = int(query_prop * len(df))
df['split'] = 'database'
df.loc[np.random.choice(df.index, query_size, replace=False), 'split'] = 'query'

# Save the DataFrame to a CSV file
save_path = os.path.join(directory, f'{directory_name}.csv')
df.to_csv(save_path, index=False)

