import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from data_prep import split_dataset, augment_dataset
from convert_voc_to_yolo import main as voc2yolo_main


def download_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dst


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    return out_dir


def _find_voc_root(root: Path) -> Path | None:
    # Look for a directory containing JPEGImages and Annotations, search up to depth 3
    candidates = [root]
    try:
        candidates += [p for p in root.iterdir() if p.is_dir()]
        for p in list(candidates[1:]):
            candidates += [q for q in p.iterdir() if q.is_dir()]
    except Exception:
        pass
    for base in candidates:
        if (base / "JPEGImages").is_dir() and (base / "Annotations").is_dir():
            return base
    return None


def convert_voc_to_yolo(voc_root: Path, out_raw: Path) -> None:
    # try to locate VOC structure
    detected_root = _find_voc_root(voc_root)
    if detected_root is None:
        raise RuntimeError("Cannot locate VOC images/annotations folders. Ensure a folder contains 'JPEGImages' and 'Annotations'.")
    images = detected_root / "JPEGImages"
    annots = detected_root / "Annotations"

    # call converter entry with args
    import sys
    sys.argv = [
        "convert_voc_to_yolo",
        "--voc-images", str(images),
        "--voc-annots", str(annots),
        "--out", str(out_raw),
    ]
    voc2yolo_main()


def main():
    parser = argparse.ArgumentParser(description="Download/prepare SeaShips-like VOC dataset and produce YOLO training set")
    parser.add_argument("--source", required=True, help="URL to ZIP or local ZIP/folder of VOC dataset")
    parser.add_argument("--workdir", type=Path, default=Path("datasets"))
    parser.add_argument("--raw-out", type=Path, default=Path("dataset_raw"))
    parser.add_argument("--final-out", type=Path, default=Path("dataset_yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-per-image", type=int, default=0)
    args = parser.parse_args()

    src = args.source
    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)

    # Resolve source into local folder path containing VOC structure
    local_root: Path
    if src.startswith("http://") or src.startswith("https://"):
        zip_name = Path(urlparse(src).path).name or "dataset.zip"
        zip_path = work / zip_name
        print(f"Downloading dataset from {src} -> {zip_path}")
        download_file(src, zip_path)
        extract_dir = work / zip_name.replace(".zip", "")
        print(f"Extracting {zip_path} -> {extract_dir}")
        extract_zip(zip_path, extract_dir)
        local_root = extract_dir
    else:
        local_path = Path(src)
        if local_path.is_file() and local_path.suffix.lower() == ".zip":
            extract_dir = work / local_path.stem
            print(f"Extracting {local_path} -> {extract_dir}")
            extract_zip(local_path, extract_dir)
            local_root = extract_dir
        else:
            local_root = local_path

    # Convert to YOLO raw
    # If somehow a .zip path slipped through, extract it now
    if isinstance(local_root, Path) and local_root.is_file() and local_root.suffix.lower() == ".zip":
        extract_dir = work / local_root.stem
        print(f"Extracting {local_root} -> {extract_dir}")
        extract_zip(local_root, extract_dir)
        local_root = extract_dir

    print(f"Converting VOC -> YOLO from {local_root}")
    if args.raw_out.exists():
        shutil.rmtree(args.raw_out)
    convert_voc_to_yolo(local_root, args.raw_out)

    # Split and augment
    if args.final_out.exists():
        shutil.rmtree(args.final_out)
    split_dataset(args.raw_out, args.final_out, args.val_ratio, args.test_ratio)
    if args.augment and args.aug_per_image > 0:
        augment_dataset(args.final_out, args.aug_per_image)

    print("Prepared YOLO dataset at:", args.final_out)


if __name__ == "__main__":
    main()


