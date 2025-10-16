import argparse
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def copy_with_dirs(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def horizontal_flip(image, labels):
    h, w = image.shape[:2]
    flipped = cv2.flip(image, 1)
    new_labels = []
    for cls, x, y, bw, bh in labels:
        new_labels.append((cls, 1.0 - x, y, bw, bh))
    return flipped, new_labels


def read_yolo_labels(label_path: Path):
    if not label_path.exists():
        return []
    items = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                items.append((cls, x, y, w, h))
    return items


def write_yolo_labels(label_path: Path, labels) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def split_dataset(raw_dir: Path, out_dir: Path, val_ratio: float, test_ratio: float):
    img_dir = raw_dir / "images"
    lbl_dir = raw_dir / "labels"
    images = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    random.shuffle(images)
    n = len(images)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[: n - n_val - n_test],
        "val": images[n - n_val - n_test : n - n_test],
        "test": images[n - n_test :],
    }

    for split_name, split_images in splits.items():
        for img_path in tqdm(split_images, desc=f"Copy {split_name}"):
            rel = img_path.relative_to(img_dir)
            dst_img = out_dir / "images" / split_name / rel
            copy_with_dirs(img_path, dst_img)
            lbl_rel = rel.with_suffix(".txt")
            src_lbl = lbl_dir / lbl_rel
            dst_lbl = out_dir / "labels" / split_name / lbl_rel
            if src_lbl.exists():
                copy_with_dirs(src_lbl, dst_lbl)


def augment_dataset(out_dir: Path, aug_per_image: int):
    train_images = list((out_dir / "images" / "train").rglob("*.jpg")) + list(
        (out_dir / "images" / "train").rglob("*.png")
    )
    for img_path in tqdm(train_images, desc="Augment"):
        label_path = out_dir / "labels" / "train" / img_path.relative_to(
            out_dir / "images" / "train"
        ).with_suffix(".txt")
        image = cv2.imread(str(img_path))
        labels = read_yolo_labels(label_path)
        for k in range(aug_per_image):
            aug_img, aug_labels = horizontal_flip(image, labels) if k % 2 == 0 else (image, labels)
            aug_name = img_path.stem + f"_aug{k}"
            dst_img = img_path.with_name(aug_name + img_path.suffix)
            dst_lbl = label_path.with_name(aug_name + ".txt")
            cv2.imwrite(str(dst_img), aug_img)
            write_yolo_labels(dst_lbl, aug_labels)


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset with splits and simple augmentations")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-per-image", type=int, default=0)
    args = parser.parse_args()

    random.seed(42)
    if args.out_dir.exists():
        print(f"Output dir {args.out_dir} exists. New files may be added alongside existing ones.")
    split_dataset(args.raw_dir, args.out_dir, args.val_ratio, args.test_ratio)
    if args.augment and args.aug_per_image > 0:
        augment_dataset(args.out_dir, args.aug_per_image)


if __name__ == "__main__":
    main()


