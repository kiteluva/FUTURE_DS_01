import kagglehub
import os
import pandas as pd

dataset_root_path = kagglehub.dataset_download("mohamed38/superstoredataset")
print(f"Dataset is located at: {dataset_root_path}")

# List files in the downloaded dataset directory
print("Files in the dataset directory:")
for root, dirs, files in os.walk(dataset_root_path):
    for file in files:
        print(os.path.join(root, file))