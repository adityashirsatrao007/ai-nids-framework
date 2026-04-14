import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """
    Downloads a file with a progress bar.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong during download")
        return False
    return True

def main():
    # Official mirror for MachineLearningCSV.zip
    url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"
    zip_path = "dataset/MachineLearningCSV.zip"
    extract_path = "dataset/"
    
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
        
    print(f"Checking for dataset at {zip_path}...")
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {url}...")
        try:
            success = download_file(url, zip_path)
            if not success:
                print("Download failed. Please download manually from UNB website.")
                return
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure you have an active internet connection or download manually.")
            return
    else:
        print("Dataset zip already exists.")

    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Dataset extracted to {extract_path}")
    except zipfile.BadZipFile:
        print("The downloaded file is corrupted. Please delete it and try again.")

if __name__ == "__main__":
    main()
