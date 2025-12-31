# 01.CheckDataset.py
import yaml
import os
from config import DATASET_YAML

def check_dataset():
    if not os.path.exists(DATASET_YAML):
        print(f"Error: {DATASET_YAML} not found. Please download from Roboflow.")
        return False
    
    with open(DATASET_YAML, 'r') as f:
        data = yaml.safe_load(f)
        
    print("--- Dataset Info ---")
    print(f"Classes: {data['names']}")
    print(f"Train Path: {data['train']}")
    print(f"Validation Path: {data['val']}")
    return True

if __name__ == "__main__":
    check_dataset()