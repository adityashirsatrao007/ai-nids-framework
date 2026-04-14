"""
Upload large research artifacts (models, data) to Hugging Face Dataset repo.
Skips files that are already present on the Hub.

Usage:
    Set HF_TOKEN environment variable, then run:
    python demo/hf_upload_data.py
"""
import os
import sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

TOKEN = os.environ.get("HF_TOKEN")
USERNAME = "adityashirsatrao007"
DATASET_REPO_ID = f"{USERNAME}/AI-NIDS-Data"

# All large files to back up on Hugging Face
LARGE_FILES = [
    "results/ready_sample.csv",
    "results/ensemble_v1.joblib",
    "results/explainer.joblib",
    "results/cnn_lstm_v1.pth",
    "results/scaler.joblib",
    "results/selected_features.joblib",
    "results/feature_names.joblib",
    "results/ablation_results.csv",
    "results/ensemble_metrics.csv",
]

def file_exists_on_hub(api, repo_id, path_in_repo):
    """Check if a file already exists in the HF dataset repo."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=TOKEN,
            local_dir="/tmp/hf_check"
        )
        return True
    except EntryNotFoundError:
        return False
    except Exception:
        return False

def upload_large_files():
    if not TOKEN:
        print("[-] HF_TOKEN not set. Run: $env:HF_TOKEN = 'your_token'")
        sys.exit(1)

    api = HfApi(token=TOKEN)
    print(f"[*] Authenticated as: {USERNAME}")

    # Create Dataset repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"[+] Dataset repo ready: {DATASET_REPO_ID}")
    except Exception as e:
        print(f"[-] Repo creation error: {e}")
        sys.exit(1)

    skipped, uploaded = 0, 0

    for local_path in LARGE_FILES:
        if not os.path.exists(local_path):
            print(f"[~] Skipping (not found locally): {local_path}")
            continue

        # Check if already on Hub
        if file_exists_on_hub(api, DATASET_REPO_ID, local_path):
            print(f"[=] Already on Hub, skipping: {local_path}")
            skipped += 1
            continue

        # Upload
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[^] Uploading {local_path} ({size_mb:.1f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=local_path,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload: {os.path.basename(local_path)}"
            )
            print(f"[+] Done: {local_path}")
            uploaded += 1
        except Exception as e:
            print(f"[-] Failed to upload {local_path}: {e}")

    print(f"\n[SUMMARY] Uploaded: {uploaded} | Skipped (already present): {skipped}")
    print(f"Dataset URL: https://huggingface.co/datasets/{DATASET_REPO_ID}")

if __name__ == "__main__":
    upload_large_files()
