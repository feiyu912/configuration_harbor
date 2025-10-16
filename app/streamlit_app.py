import argparse
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image


def run(images_dir: Path, preds_csv: Path):
    st.set_page_config(page_title="Port Detection Dashboard", layout="wide")
    st.title("Port Detection Dashboard")

    if not preds_csv.exists():
        st.error(f"Predictions CSV not found: {preds_csv}")
        return
    df = pd.read_csv(preds_csv)
    class_map = {0: "ship", 1: "container", 2: "crane"}
    df["class"] = df["class_id"].map(class_map).fillna(df["class_id"].astype(str))

    st.sidebar.header("Filters")
    classes = st.sidebar.multiselect("Classes", options=sorted(df["class"].unique()), default=list(sorted(df["class"].unique())))
    conf_min = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.25, 0.01)

    fdf = df[(df["class"].isin(classes)) & (df["score"] >= conf_min)]
    st.dataframe(fdf.head(200))

    image_names = sorted(fdf["image"].unique())
    for name in image_names:
        img_path = next((p for p in (images_dir).rglob(name)), None)
        if not img_path:
            continue
        st.subheader(name)
        st.image(Image.open(img_path), use_column_width=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--preds-csv", required=True)
    args = parser.parse_args()
    run(Path(args.images_dir), Path(args.preds_csv))


if __name__ == "__main__":
    main()


