import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_USER = "hamedhani"
DATASET_REF = f"{KAGGLE_USER}/alignlab-qlora-adapter"
ADAPTER_REPO = "hamedready/alignlab-qwen3-1.7b-qlora"
KERNEL_ID = f"{KAGGLE_USER}/02-alignlab-model-evaluation"

STAGE_DIR = Path("kaggle/adapter-stage")
RUN_DIR = Path("kaggle/eval-run")
SOURCE_METADATA = Path("kaggle/evaluation/kernel-metadata.json")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def dataset_exists(api: KaggleApi, dataset_ref: str) -> bool:
    """Check whether the Kaggle dataset exists."""
    username, slug = dataset_ref.split("/", maxsplit=1)

    datasets = api.dataset_list(
        user=username,
        search=slug,
    )

    return any(getattr(dataset, "ref", None) == dataset_ref for dataset in datasets)


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN must be set in GitHub Actions secrets")

    api_hf = HfApi(token=hf_token)
    print("HF authenticated as:", api_hf.whoami()["name"])

    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)

    snapshot_download(
        repo_id=ADAPTER_REPO,
        repo_type="model",
        token=hf_token,
        local_dir=str(STAGE_DIR),
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
            "generation_config.json",
        ],
    )

    if not (STAGE_DIR / "adapter_config.json").is_file():
        raise FileNotFoundError("Missing adapter_config.json in Hugging Face adapter")
    if not any(
        (STAGE_DIR / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError("Missing adapter_model.safetensors or adapter_model.bin")

    dataset_metadata = {
        "title": "AlignLab QLoRA Adapter (Private Evaluation Input)",
        "id": DATASET_REF,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }
    (STAGE_DIR / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n", encoding="utf-8"
    )

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    if dataset_exists(kaggle_api, DATASET_REF):
        print("Updating existing Kaggle dataset:", DATASET_REF)
        run_command(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(STAGE_DIR),
                "-m",
                "Update QLoRA adapter for evaluation",
                "--dir-mode",
                "skip",
            ]
        )
    else:
        print("Creating private Kaggle dataset:", DATASET_REF)
        run_command(["kaggle", "datasets", "create", "-p", str(STAGE_DIR), "--dir-mode", "skip"])

    if not SOURCE_METADATA.is_file():
        raise FileNotFoundError(f"Missing evaluation metadata: {SOURCE_METADATA}")

    kernel_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    if kernel_metadata.get("id") != KERNEL_ID:
        raise ValueError(f"Expected kernel id {KERNEL_ID!r}")
    if kernel_metadata.get("code_file") != "02_model_evaluation.ipynb":
        raise ValueError("Evaluation code_file must be 02_model_evaluation.ipynb")

    kernel_metadata["dataset_sources"] = [DATASET_REF]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUN_DIR / "kernel-metadata.json"
    output_path.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")

    print("Staged adapter dataset:", DATASET_REF)
    print("Prepared evaluation metadata:", output_path)
    print("Dataset sources:", kernel_metadata["dataset_sources"])


if __name__ == "__main__":
    main()
