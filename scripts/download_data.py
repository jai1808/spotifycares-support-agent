"""
Script to download the Kaggle 'Customer Support on Twitter' dataset.
Prefers kagglehub, then Kaggle CLI, then existing local file.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "twcs.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1_000_000:
        print(f"Dataset already exists at {csv_path}")
        return

    # 1) kagglehub (handles auth/cache)
    try:
        import kagglehub

        print("Downloading via kagglehub...")
        cache_path = Path(
            kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        )
        candidates = list(cache_path.rglob("twcs.csv"))
        if candidates:
            shutil.copy2(candidates[0], csv_path)
            print(f"Copied to {csv_path}")
            return
    except Exception as e:
        print(f"kagglehub failed: {e}")

    # 2) Kaggle CLI
    try:
        print("Attempting download via Kaggle CLI...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "thoughtvector/customer-support-on-twitter",
                "-p",
                str(data_dir),
            ],
            check=True,
        )
        import zipfile

        for zip_path in data_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            zip_path.unlink(missing_ok=True)
        if csv_path.exists():
            print("Download and extraction complete.")
            return
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Kaggle CLI failed: {e}")

    print(
        "\nCould not download automatically.\n"
        "Install kagglehub (`pip install kagglehub`) and authenticate, or place twcs.csv at:\n"
        f"  {csv_path}"
    )


if __name__ == "__main__":
    main()
