import torch
from typing import List, Union
import pandas as pd

import utils.augmenter.base




class AugmentDataset(torch.utils.data.Dataset):
    """
    データオーギュメンテーションもできるようにしたDataset

    Parameter
    ---------
    phase: str -> ["train", "val"]
        訓練モードか検証モード
    """

    def __init__(self, dataframe: pd.DataFrame, lookups: List[str], tokenizer, phase: str, max_len: int, augmenter: utils.augmenter.base.AugmenterWithNER or None = None) -> None:
        self.data = dataframe
        self.labels_to_ids = {v:k for k, v in enumerate(lookups)}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.phase = phase    # for validation
        self.augmenter = augmenter

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.pull_item(index)
        return item

    def pull_item(self, index):
        # テキストとラベルを取得する
        text = self.data.text[index]
        word_labels = self.data.annotation[index] if self.phase == "train" else None

        # DataAugment
        if self.augmenter and self.phase == "train":
            try:
                aug_text, aug_label = self.augmenter.augment(text, word_labels)
            except:
                text = text.lower()
                aug_text, aug_label = self.augmenter.augment(text, word_labels)
        else:
            aug_text, aug_label = text, word_labels

        # tokenize
        encoding = self.tokenizer(aug_text.split(), is_split_into_words=True, padding="max_length", truncation=True, max_length=self.max_len)
        word_ids = encoding.word_ids()

        # ターゲットを作る
        if self.phase == "train":
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.labels_to_ids[aug_label[word_idx]])
                else:
                    label_ids.append(self.labels_to_ids[aug_label[word_idx]])
                previous_word_idx = word_idx
            encoding["labels"] = label_ids
        
        # torchTensorに変換
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.phase == "val":
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item["wids"] = torch.as_tensor(word_ids2)
        return item