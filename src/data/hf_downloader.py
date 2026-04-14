from datasets import load_dataset
import pandas as pd
import os

def download_real_dataset():
    """
    Downloads the real CICIDS2017 dataset from a verified HuggingFace mirror.
    Saves it as a CSV for compatibility with the existing pipeline.
    """
    print("--- Fetching Real CICIDS2017 Dataset from HuggingFace ---")
    try:
        # Using a balanced and verified dataset on HF
        # This dataset contains the original features and labels
        dataset = load_dataset("c01dsnap/CIC-IDS2017", split="train")
        
        print("Converting to Pandas for processing...")
        df = dataset.to_pandas()
        
        # Ensure the dataset directory exists
        os.makedirs("dataset/MachineLearningCSV", exist_ok=True)
        
        output_path = "dataset/MachineLearningCSV/real_data_hf.csv"
        print(f"Saving real dataset to {output_path} (this may take a moment)...")
        df.to_csv(output_path, index=False)
        
        print(f"Real dataset successfully acquired: {df.shape[0]} rows, {df.shape[1]} features.")
        return output_path
        
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        return None

if __name__ == "__main__":
    download_real_dataset()
