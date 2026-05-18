"""
Download Pythia models from HuggingFace into /mnt/xdata/llm_weights/models/.
Only downloads model weights + config, skipping tokenizer/optimizer states.
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

# Only download what we need for weight extraction
INCLUDE_PATTERNS = ["*.safetensors", "*.json"]
IGNORE_PATTERNS = ["optimizer*", "scheduler*", "trainer_state*", "tokenizer*"]


def download_model(repo_id: str, base_dir: Path) -> Path:
    model_name = repo_id.split("/")[-1]
    local_dir = base_dir / model_name
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=INCLUDE_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"  Done: {repo_id}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download Pythia models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/mnt/xdata/llm_weights/models"),
        help="Directory to download models into",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help="Which sizes to download, e.g. --sizes 70m 160m 12b. Default: all.",
    )
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    models = PYTHIA_MODELS
    if args.sizes:
        sizes = set(args.sizes)
        models = [m for m in PYTHIA_MODELS if m.split("-")[-1] in sizes]
        if not models:
            print(f"No models matched sizes: {args.sizes}")
            return

    print(f"Downloading {len(models)} model(s) to {args.models_dir}\n")
    for repo_id in models:
        download_model(repo_id, args.models_dir)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
