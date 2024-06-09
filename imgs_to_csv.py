import os
import pandas as pd
import numpy as np
from PIL import Image

# Define the directory containing the JPG files
directory_name = 'dogs'
directory = f'./{directory_name}'

# Function to check if an image is corrupted
def is_corrupted(filepath):
    try:
        img = Image.open(filepath)  # Open the image file
        img.verify()  # Verify that it is, indeed, an image
        return False
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image: {filepath}")
        return True

# List all JPG files in the directory
files = [os.path.join(directory, file) for file in os.listdir(directory) 
         if file.endswith('.jpeg') and not is_corrupted(os.path.join(directory, file))]

# Create a DataFrame with file paths
df = pd.DataFrame(files, columns=['path'])

# Assign 'split' column
query_prop = 0.2
query_size = int(query_prop * len(df))
df['split'] = 'database'
df.loc[np.random.choice(df.index, query_size, replace=False), 'split'] = 'query'

# Save the DataFrame to a CSV file
df.to_csv(f'{directory}/{directory_name}.csv', index=False)
