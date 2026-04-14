import os
import sys
from huggingface_hub import HfApi

# Configuration
TOKEN = os.environ.get("HF_TOKEN")
USERNAME = "adityashirsatrao007"
REPO_NAME = "AI-NIDS-Research-Framework"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

def sync_to_hf():
    if not TOKEN:
        print("[-] HF_TOKEN environment variable not set.")
        sys.exit(1)
        
    api = HfApi(token=TOKEN)
    
    print(f"[*] Authenticated as: {USERNAME}")
    print(f"[*] Target Repo: {REPO_ID} (Space/Docker)")
    
    # Create the Space if it doesn't exist
    try:
        api.create_repo(
            repo_id=REPO_ID, 
            repo_type="space", 
            space_sdk="docker", 
            private=False, 
            exist_ok=True
        )
        print("[+] Repository verified/created.")
    except Exception as e:
        print(f"[-] Repo creation error: {e}")

    # Define high-importance files (Models, Paper, Data)
    print("[*] Uploading core research assets (including large files)...")
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=REPO_ID,
            repo_type="space",
            ignore_patterns=[
                "venv/*", ".git/*", "__pycache__/*", "*.pyc",
                "dataset/*", 
                "demo/hf_sync_final.py", "demo/deploy_hf.py"
            ],
            commit_message="Full Research Bundle: Code, Models, Sample Data, and IEEE Manuscript"
        )
        print(f"\n[SUCCESS] All research assets are now SECURE on Hugging Face!")
        print(f"Live Project URL: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"[-] Final sync failed: {e}")

if __name__ == "__main__":
    sync_to_hf()
