#!/usr/local/bin/python
"""Check the CC120K layout and the runtime used by this project."""

import argparse
from pathlib import Path
import sys


def read_ids(path):
    return path.read_text(encoding="utf-8").splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/data/CC120K")
    parser.add_argument("--noise_ratio", type=float, default=0)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    split = root / "annotations" / "scan_split"
    images = root / "images"
    errors = []

    for path in [split, images, root / "annotations" / "memory_bank"]:
        if not path.exists():
            errors.append(f"missing: {path}")

    expected = {"train": 118851, "dev": 1000, "test": 1000}
    all_ids = set()
    for name, count in expected.items():
        ids_path = split / f"{name}_ids.txt"
        caps_path = split / f"{name}_caps.txt"
        if not ids_path.exists() or not caps_path.exists():
            errors.append(f"missing split files for {name}")
            continue
        ids = read_ids(ids_path)
        caps = read_ids(caps_path)
        print(f"{name}: ids={len(ids)} captions={len(caps)}")
        if len(ids) != count or len(caps) != count:
            errors.append(f"{name} expected {count} rows, got ids={len(ids)}, captions={len(caps)}")
        if len(ids) != len(caps):
            errors.append(f"{name} ids/captions length mismatch")
        all_ids.update(ids)

    if images.exists():
        image_ids = {path.stem for path in images.glob("*.jpg")}
        missing = sorted(all_ids - image_ids)
        print(f"images: files={len(image_ids)} referenced={len(all_ids)} missing={len(missing)}")
        if missing:
            errors.append(f"{len(missing)} referenced images are missing")

    ratio_name = "0" if args.noise_ratio == 0 else str(args.noise_ratio)
    for suffix in ["mbank_img_idx.npy", "mbank_txt_idx.npy"]:
        path = root / "annotations" / "memory_bank" / f"{ratio_name}_{suffix}"
        if not path.exists():
            errors.append(f"missing memory bank: {path}")
        else:
            print(f"memory bank: {path.name}")

    try:
        import torch
        import torchvision
        import ftfy  # noqa: F401
        print(f"python: {sys.executable}")
        print(f"torch: {torch.__version__}, torchvision: {torchvision.__version__}")
        print(f"cuda: built={torch.version.cuda}, available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        errors.append(f"runtime import failed: {exc}")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("CC120K setup OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
