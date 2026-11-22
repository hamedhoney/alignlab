import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

KAGGLE_USER = "hamedhani"
DATASET_SLUG = "alignlab-qlora-adapter"
DATASET_REF = f"{KAGGLE_USER}/{DATASET_SLUG}"

ADAPTER_REPO = "hamedready/alignlab-qwen3-1.7b-qlora"


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    token = os.environ["HF_TOKEN"]

    # --------------------------------------------------
    # 1. Prepare clean staging directory
    # --------------------------------------------------

    stage = Path("kaggle/adapter-stage")

    if stage.exists():
        shutil.rmtree(stage)

    stage.mkdir(parents=True)

    # --------------------------------------------------
    # 2. Download adapter from Hugging Face
    # --------------------------------------------------

    api = HfApi(token=token)

    print("HF authenticated as:", api.whoami()["name"])

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
        raise RuntimeError("Adapter config missing")

    weights = [
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]

    if not any((stage / name).is_file() for name in weights):
        raise RuntimeError("Adapter weights missing")

    # --------------------------------------------------
    # 3. Create Kaggle dataset metadata
    # --------------------------------------------------

    dataset_metadata = {
        "title": "AlignLab QLoRA Adapter",
        "id": DATASET_REF,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }

    (stage / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 4. Create or update Kaggle dataset
    # --------------------------------------------------

    check = subprocess.run(
        [
            "kaggle",
            "datasets",
            "metadata",
            DATASET_REF,
            "-p",
            str(stage),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if check.returncode == 0:
        print("Updating existing Kaggle dataset")

        run_command(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(stage),
                "-m",
                "Update QLoRA adapter",
                "--dir-mode",
                "skip",
            ]
        )

    elif "404" in (check.stdout + check.stderr):
        print("Creating new Kaggle dataset")

        run_command(
            [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(stage),
                "--dir-mode",
                "skip",
            ]
        )

    else:
        raise RuntimeError("Could not check Kaggle dataset: " + check.stderr)

    print("Kaggle adapter dataset:", DATASET_REF)

    # --------------------------------------------------
    # 5. Prepare evaluation notebook metadata
    # --------------------------------------------------

    metadata_path = Path("kaggle/evaluation/kernel-metadata.json")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata["dataset_sources"] = [DATASET_REF]

    run_dir = Path("kaggle/eval-run")
    run_dir.mkdir(parents=True, exist_ok=True)

    output_metadata = run_dir / "kernel-metadata.json"

    output_metadata.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Evaluation metadata prepared:", output_metadata)


if __name__ == "__main__":
    main()
