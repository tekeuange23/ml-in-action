import os
from datasets import load_dataset

os.environ['HF_HUB_OFFLINE'] = '1'


# Load the dataset
data = load_dataset(
  'ajinkyakolhe112/cats_vs_dogs_classification_kaggle', 
  cache_dir='datasets/cats_vs_dogs_classification',
) 

print(data)
for key in data.keys():
    print(f'{key}: {len(data[key])}')
    for i in range(3):
        print(f'{data[key][i]}')
    print()