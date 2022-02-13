import torch 

class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe, lookups, tokenizer, max_len, val):
        self.data = dataframe
        self.labels_to_ids = {v:k for k, v in enumerate(lookups)}
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.val = val    # for validation
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.pull_item(index)
        return item

    def pull_item(self, index):
        # テキストとラベルを取得する
        text = self.data.text[index]
        word_labels = self.data.annotation[index] if not self.val else None

        # tokenize
        encoding = self.tokenizer(text.split(), is_split_into_words=True, padding="max_length", truncation=True, max_length=self.max_len)
        word_ids = encoding.word_ids()

        # ターゲットを作る
        if not self.val:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.labels_to_ids[word_labels[word_idx]])
                else:
                    label_ids.append(self.labels_to_ids[word_labels[word_idx]])
                previous_word_idx = word_idx
            encoding["labels"] = label_ids
        
        # torchTensorに変換
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.val:
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item["wids"] = torch.as_tensor(word_ids2)
        return item