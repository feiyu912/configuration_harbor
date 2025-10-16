import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        bbox = obj.find("bndbox")
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))
        objects.append((name, xmin, ymin, xmax, ymax))
    return width, height, objects


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = (xmax - xmin)
    h = (ymax - ymin)
    return x / img_w, y / img_h, w / img_w, h / img_h


def main():
    parser = argparse.ArgumentParser(description="Convert VOC annotations to YOLO for three classes")
    parser.add_argument("--voc-images", type=Path, required=True)
    parser.add_argument("--voc-annots", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("dataset_raw"))
    parser.add_argument("--class-map", nargs="+", default=["ship", "container", "crane"], help="ordered class names")
    parser.add_argument("--force-class", type=str, default=None, help="force map ALL objects to this class name (e.g., 'ship')")
    parser.add_argument("--accept-names", nargs="*", default=[], help="additional accepted object names to map to existing classes")
    args = parser.parse_args()

    img_out = args.out / "images"
    lbl_out = args.out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: idx for idx, name in enumerate(args.class_map)}

    xml_files = sorted(args.voc_annots.glob("*.xml"))
    kept_boxes = 0
    for xml_path in tqdm(xml_files, desc="Convert VOC"):
        stem = xml_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = args.voc_images / (stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        w, h, objects = parse_xml(xml_path)
        # Fallback to reading image size when VOC size is missing
        if w == 0 or h == 0:
            with Image.open(img_path) as im:
                w, h = im.size

        lines = []
        for name, xmin, ymin, xmax, ymax in objects:
            if args.force_class:
                lname = args.force_class.strip().lower()
            else:
                lname = name.strip().lower()
                # simple normalization of synonyms
                if lname in {"ship", "boat", "vessel", "ships"}:
                    lname = "ship"
                elif lname in {"container", "shipping container", "containers"}:
                    lname = "container"
                elif lname in {"crane", "gantry crane", "tower crane", "quay crane", "port crane"}:
                    lname = "crane"
                # allow user-provided accepted names; map to closest if exact key exists
                if lname not in class_to_id and args.accept_names:
                    if lname in args.accept_names:
                        # default to first class when ambiguous
                        lname = args.class_map[0]
            if lname not in class_to_id:
                continue
            x, y, bw, bh = voc_box_to_yolo(xmin, ymin, xmax, ymax, w, h)
            lines.append(f"{class_to_id[lname]} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
            kept_boxes += 1

        if not lines:
            continue
        # copy image and write label
        img_dst = img_out / img_path.name
        if not img_dst.exists():
            img_dst.write_bytes(img_path.read_bytes())
        (lbl_out / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")

    print(f"Kept {kept_boxes} boxes after mapping. Output: {lbl_out}")


if __name__ == "__main__":
    main()


