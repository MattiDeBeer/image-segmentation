from huggingface_hub import HfApi, Repository, login
import os

REPO_NAME = "mattidebeer/clip-unet-model"
LOCAL_DIR = "./"

# Login
login()

# Create repo if it doesn't exist
api = HfApi()
api.create_repo(repo_id=REPO_NAME, exist_ok=True)

# Push everything
repo = Repository(local_dir=LOCAL_DIR, clone_from=REPO_NAME)
repo.push_to_hub()
