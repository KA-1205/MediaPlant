"""
scripts/prepare_data.py
========================
Scans data/processed/ and creates stratified train/val/test split CSVs.

Usage:
    python scripts/prepare_data.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    data_dir = "data/processed"
    out_dir  = "data/splits"
    os.makedirs(out_dir, exist_ok=True)

    records = []
    classes = sorted(os.listdir(data_dir))
    c2i     = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path): continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({"image_path": os.path.join(cls_path, fname),
                                 "class_name": cls, "label": c2i[cls]})

    df = pd.DataFrame(records)
    print(f"Total: {len(df)} images | {len(classes)} classes")

    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test   = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(  f"{out_dir}/val.csv",   index=False)
    test.to_csv( f"{out_dir}/test.csv",  index=False)
    pd.DataFrame(list(c2i.items()), columns=["class_name","label"]).to_csv(
        f"{out_dir}/class_map.csv", index=False)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
