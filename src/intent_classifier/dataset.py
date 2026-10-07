import pandas as pd
import torch
from torch.utils.data import Dataset

class IntentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["text", "label"])
        self.df["label"] = self.df["label"].astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # token_type_ids 없으면 직접 생성
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))

        # KoBERT 안전 처리: 0 또는 1만 남기기
        token_type_ids = torch.clamp(token_type_ids, min=0, max=1)

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item
