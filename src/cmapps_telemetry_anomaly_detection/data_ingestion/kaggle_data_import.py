import subprocess
import shutil
from pathlib import Path
import yaml


def _has_real_files(folder: Path) -> bool:
    """
    Returns True if folder contains any files other than placeholders like .gitkeep.
    """
    for p in folder.iterdir():
        # ignore placeholders / hidden files
        if p.name in {".gitkeep", ".DS_Store"}:
            continue
        if p.name.startswith("."):
            continue
        return True
    return False


def download_dataset(config_path: str = "configs/data.yaml", force: bool = False) -> Path:
    """
    Download dataset from Kaggle into data/raw directory.
    Returns the path to the raw data folder.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_slug = config["dataset"]["slug"]
    raw_path = Path(config["dataset"]["raw_path"]).resolve()
    raw_path.mkdir(parents=True, exist_ok=True)

    if not force and _has_real_files(raw_path):
        print(f"[INFO] Data already exists in {raw_path}")
        return raw_path

    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI not found. Install it with: pip install kaggle")

    print(f"[INFO] Downloading dataset: {dataset_slug}")
    print(f"[INFO] Saving to: {raw_path}")

    command = [
        "kaggle",
        "datasets",
        "download",
        dataset_slug,
        "-p",
        str(raw_path),
        "--unzip",
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")

    print("[SUCCESS] Dataset downloaded successfully.")
    return raw_path