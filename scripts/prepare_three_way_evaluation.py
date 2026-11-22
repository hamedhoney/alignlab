"""Stage the private DPO adapter for Kaggle and prepare Experiment 04 metadata."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_REF = "hamedhani/alignlab-dpo-adapter"
DPO_REPO = "hamedready/alignlab-qwen3-1.7b-dpo"
RUN_DIR = Path("kaggle/three-way-run")
STAGE_DIR = Path("kaggle/dpo-adapter-stage")


def main() -> None:
    token = os.environ["HF_TOKEN"].strip()
    if not token:
        raise RuntimeError("HF_TOKEN is empty")
    print("Hugging Face user:", HfApi(token=token).whoami()["name"])
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)
    snapshot_download(
        repo_id=DPO_REPO,
        token=token,
        local_dir=str(STAGE_DIR),
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )
    if not (STAGE_DIR / "adapter_config.json").is_file():
        raise FileNotFoundError("DPO adapter_config.json missing")
    if not any(
        (STAGE_DIR / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError("DPO adapter weights missing")

    (STAGE_DIR / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": DATASET_REF,
                "title": "AlignLab DPO Adapter",
                "licenses": [{"name": "unknown"}],
                "isPrivate": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    api = KaggleApi()
    api.authenticate()
    datasets = api.dataset_list(user="hamedhani", search="alignlab-dpo-adapter")
    exists = any(getattr(item, "ref", None) == DATASET_REF for item in datasets)
    if exists:
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(STAGE_DIR),
            "-m",
            "Refresh DPO adapter",
            "--dir-mode",
            "skip",
        ]
    else:
        command = ["kaggle", "datasets", "create", "-p", str(STAGE_DIR), "--dir-mode", "skip"]
    subprocess.run(command, check=True)
    print("DPO dataset staged:", DATASET_REF)
    # Verify the uploaded dataset is visible before submitting a GPU notebook.
    refreshed = api.dataset_list(user="hamedhani", search="alignlab-dpo-adapter")
    if not any(getattr(item, "ref", None) == DATASET_REF for item in refreshed):
        raise RuntimeError(f"Kaggle dataset is not visible after upload: {DATASET_REF}")

    metadata = json.loads(Path("kaggle/three-way/kernel-metadata.json").read_text())
    expected = {"hamedhani/alignlab-qlora-adapter", DATASET_REF}
    if set(metadata["dataset_sources"]) != expected:
        raise ValueError("Experiment 04 must attach both SFT and DPO datasets")
    if metadata.get("id") != "hamedhani/04-alignlab-three-way-evaluation":
        raise ValueError("Unexpected Kaggle kernel ID")
    if metadata.get("code_file") != "04_three_way_evaluation.ipynb":
        raise ValueError("Unexpected Kaggle code_file")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Notebook metadata prepared:", RUN_DIR / "kernel-metadata.json")
    print("Dataset sources:", metadata["dataset_sources"])


if __name__ == "__main__":
    main()
