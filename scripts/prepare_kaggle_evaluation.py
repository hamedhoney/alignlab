import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

KAGGLE_USER = "hamedhani"
DATASET_SLUG = "alignlab-qlora-adapter"
DATASET_REF = f"{KAGGLE_USER}/{DATASET_SLUG}"
ADAPTER_REPO = "hamedready/alignlab-qwen3-1.7b-qlora"


def main() -> None:
    token = os.environ["HF_TOKEN"]
    stage = Path("kaggle/adapter-stage")
    stage.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=ADAPTER_REPO,
        token=token,
        local_dir=stage,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )
    if not (stage / "adapter_config.json").is_file():
        raise RuntimeError("Adapter config missing from Hugging Face download")
    if not any(
        (stage / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError("Adapter weights missing from Hugging Face download")
    (stage / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AlignLab QLoRA Adapter (Private Evaluation Input)",
                "id": DATASET_REF,
                "licenses": [{"name": "unknown"}],
                "isPrivate": True,
            },
            indent=2,
        )
    )
    api = HfApi(token=token)
    print("HF authenticated as:", api.whoami()["name"])
    # Dataset creation fails if already exists; version instead.
    check = subprocess.run(
        ["kaggle", "datasets", "status", DATASET_REF],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        cmd = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(stage),
            "-m",
            "Update adapter for evaluation",
            "--dir-mode",
            "skip",
        ]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "skip"]
    subprocess.run(cmd, check=True)
    print("Staged private Kaggle dataset:", DATASET_REF)

    run_dir = Path("kaggle/eval-run")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": f"{KAGGLE_USER}/02-alignlab-model-evaluation",
        "title": "02 AlignLab Model Evaluation",
        "code_file": "02_model_evaluation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_REF],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (run_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
