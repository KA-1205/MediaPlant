"""
Scans data/processed/ folder structure and creates train/val/test CSVs.

Expected folder structure:
    data/processed/
        PlantName1/
            img001.jpg
            img002.jpg
        PlantName2/
            ...

Usage:
    python scripts/prepare_data.py --data_dir data/processed --output_dir data/splits
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataframe(data_dir):
    records = []
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "image_path": os.path.join(cls_path, fname),
                    "class_name": cls,
                    "label": class_to_idx[cls],
                })

    df = pd.DataFrame(records)
    print(f"Total images : {len(df)}")
    print(f"Total classes: {len(classes)}")
    return df, class_to_idx


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df, class_to_idx = build_dataframe(args.data_dir)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(args.output_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(args.output_dir, "test.csv"),  index=False)

    pd.DataFrame(list(class_to_idx.items()), columns=["class_name", "label"]).to_csv(
        os.path.join(args.output_dir, "class_map.csv"), index=False
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Splits saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    args = parser.parse_args()
    main(args)
