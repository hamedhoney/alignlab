
import argparse
import json
from pathlib import Path

from datasets import load_dataset

from alignlab.inference.generate import LLMGenerator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
    )

    args = parser.parse_args()

    dataset = load_dataset(
        "openai/openai_humaneval",
        split="test",
    )

    generator = LLMGenerator(args.model)

    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "predictions.jsonl"

    with output_file.open("w") as file:
        for example in dataset.select(
            range(min(args.limit, len(dataset)))
        ):
            result = generator.generate(example["prompt"])

            record = {
                "task_id": example["task_id"],
                "prompt": example["prompt"],
                "response": result["response"],
                "latency_seconds": result["latency_seconds"],
                "generated_tokens": result["generated_tokens"],
            }

            file.write(json.dumps(record) + "\n")

            print(
                f"{record['task_id']}: "
                f"{record['latency_seconds']:.2f}s"
            )

    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()
