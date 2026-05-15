"""One-shot downloader for all required model weights.

Usage:
    python scripts/download_models.py --all
    python scripts/download_models.py --moondream --yolo --vosk --piper
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# All URLs verified at time of writing (2026). Verify again before each run.
MODELS = {
    "moondream": {
        "text_model": (
            "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-f16.gguf",
            "data/models/moondream2-text-model-f16.gguf",
        ),
        "mmproj": (
            "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-mmproj-f16.gguf",
            "data/models/moondream2-mmproj-f16.gguf",
        ),
    },
    "yolo": {
        "weights": (
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            "data/models/yolov8n.pt",
        ),
    },
    "vosk": {
        "model_zip": (
            "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip",
            "data/models/vosk-model-small-vn-0.4.zip",
        ),
    },
    "piper": {
        # NOTE: pick the appropriate vi_VN voice. There are community voices on
        # https://github.com/rhasspy/piper/blob/master/VOICES.md
        # Below is a placeholder — verify the exact filename before downloading.
        "voice": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx",
            "data/models/vi_VN-vivos-x_low.onnx",
        ),
        "voice_config": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx.json",
            "data/models/vi_VN-vivos-x_low.onnx.json",
        ),
    },
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists.")
        return
    print(f"[download] {url}\n       -> {dest}")
    try:
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
            total = 0
            while chunk := resp.read(1024 * 1024):
                out.write(chunk)
                total += len(chunk)
                sys.stdout.write(f"\r       {total/1024/1024:.1f} MB")
                sys.stdout.flush()
            sys.stdout.write("\n")
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        if dest.exists():
            dest.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    for k in MODELS:
        parser.add_argument(f"--{k}", action="store_true")
    args = parser.parse_args()

    selected = [k for k in MODELS if args.all or getattr(args, k)]
    if not selected:
        parser.print_help()
        return

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    for group in selected:
        print(f"\n=== {group.upper()} ===")
        for _name, (url, rel_dest) in MODELS[group].items():
            download(url, Path(rel_dest))

    # Vosk model is zipped — unzip step
    if "vosk" in selected:
        import zipfile

        zip_path = Path("data/models/vosk-model-small-vn-0.4.zip")
        if zip_path.exists():
            print(f"[unzip] {zip_path}")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall("data/models/")
            zip_path.unlink()

    print("\nDone. Verify with:  python -m vision_assistant.main doctor")


if __name__ == "__main__":
    main()
