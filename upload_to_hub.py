from huggingface_hub import HfApi, login

from config import HF_REPO_ID


def push_model_to_hub(model_dir: str = "./qa-mt5-final", repo_id: str = HF_REPO_ID):
    login()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)

    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
    )

    print(f"Model uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    push_model_to_hub()
