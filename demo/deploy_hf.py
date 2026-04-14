import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    import subprocess
    print("Installing huggingface_hub CLI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi

def deploy():
    # SECURE: Utilizing the token from environment variables
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[-] HF_TOKEN environment variable not set. Please set it before running.")
        sys.exit(1)
        
    api = HfApi(token=token)
    
    # Validate token
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"[+] Successfully Authenticated to HuggingFace as: {username}")
    except Exception as e:
        print(f"[-] Authentication failed: {str(e)}")
        sys.exit(1)
        
    repo_id = f"{username}/AI-NIDS-Research-Framework"
    
    print(f"[+] Registering Global Space Repository: {repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id, 
            repo_type="space", 
            space_sdk="docker", 
            private=False, 
            exist_ok=True
        )
    except Exception as e:
        print(f"[-] Registration returned: {str(e)}")
        
    print("[+] Packaging Architecture and Uploading (Compressing Heavy Data)...")
    
    # We strictly ignore raw datasets (.csv) so we fit in Cloud Free Tier and deploy instantly
    ignore_patterns = [
        "venv/*", ".git/*", "__pycache__/*", "*.csv", 
        "*.pcap", "dataset/*", "pipeline_final.log", "*.pdf", "deploy_hf.py"
    ]
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Auto-Deploying AI-NIDS Backend & Dashboards"
        )
        print("\n[SUCCESS] Enterprise Deployment Completed!")
        print(f"Live URL: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"[-] Upload failed: {str(e)}")

if __name__ == "__main__":
    deploy()
