import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(src_path: str, out_dir: str, seed: int = 42):
    """Stratified 80/10/10 split into train.csv / valid.csv / test.csv."""
    os.makedirs(out_dir, exist_ok=True)

    # 인코딩 자동 감지 (UTF-8, CP949 모두 대응)
    try:
        df = pd.read_csv(src_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(src_path, encoding="cp949")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"], shuffle=True
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"], shuffle=True
    )

    for name, part in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        part.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")

    print("dataset split done.")
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")


def remove_confusing_samples(misclassified_path: str, splits_dir: str):
    """Drop texts confused between 편견 and 비하/도발 from every split (writes *_clean.csv)."""
    wrong = pd.read_csv(misclassified_path)
    delete_texts = wrong[
        ((wrong["true_label"] == "편견") & (wrong["pred_label"] == "비하/도발")) |
        ((wrong["true_label"] == "비하/도발") & (wrong["pred_label"] == "편견"))
    ]["text"].tolist()
    print("삭제 대상 개수:", len(delete_texts))

    for name in ("train", "valid", "test"):
        df = pd.read_csv(os.path.join(splits_dir, f"{name}.csv"))
        clean = df[~df["text"].isin(delete_texts)]
        clean.to_csv(os.path.join(splits_dir, f"{name}_clean.csv"), index=False, encoding="utf-8-sig")
        print(f"{name}:", len(df), "→", len(clean))
