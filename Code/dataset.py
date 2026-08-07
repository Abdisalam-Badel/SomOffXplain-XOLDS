import pandas as pd
import torch
from torch.utils.data import Dataset

      
def preprocess_text(text):
    if pd.isna(text):
        return ""                              
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text


class AbusiveLangDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.df['text'] = self.df['text'].apply(preprocess_text)
        self.df['rationale'] = self.df['rationale'].apply(preprocess_text)

    def __len__(self):
        return len(self.df)

    def align_rationale_spans(self, text, rationale):
        if not rationale or rationale == "":
            return 0, 0

        start_char = text.find(rationale)
        if start_char == -1:
            return 0, 0
        end_char = start_char + len(rationale)

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        offsets = encoding['offset_mapping'][0]

        start_token, end_token = 0, 0
        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx
                break
        return start_token, end_token

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = int(row['label'])
        rationale = row['rationale']

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        start_idx, end_idx = self.align_rationale_spans(text, rationale)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
            'rationale_start': torch.tensor(start_idx, dtype=torch.long),
            'rationale_end': torch.tensor(end_idx, dtype=torch.long),
            'offset_mapping': encoding['offset_mapping'].squeeze(),
        }
