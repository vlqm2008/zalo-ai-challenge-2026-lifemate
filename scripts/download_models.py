#!/usr/bin/env python3
"""Pre-download model weights at Docker BUILD time (the container runs with no network).

  python scripts/download_models.py --llm Qwen/Qwen3-4B-Instruct-2507 --asr vinai/PhoWhisper-medium
"""
import argparse

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--asr", default="vinai/PhoWhisper-medium")
    args = ap.parse_args()
    for repo in (args.llm, args.asr):
        if "/" in repo and not repo.startswith(("/", ".", "models/", "out/")):
            # *.bin: PhoWhisper ships pytorch_model.bin (no safetensors); Qwen has no .bin so nothing extra is fetched
            p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "*.tiktoken", "merges.txt", "vocab.*"])
            print("downloaded", repo, "->", p)


if __name__ == "__main__":
    main()
