import os, sys, tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from pathlib import Path
from pprint import pprint
import yaml
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, load_metric

from utils.preprocessing import make_NER_dataframe, read_kfold_file
from utils.dataset import CustomDataset


def train(model, dataloader, optimizer, config):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    model.to(device)

    logs = []

    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        epoch_acc = 0.0
        iters = 0

        for g in optimizer.param_groups:
            g["lr"] = config["learning_rates"][epoch]

        phase = "train"
        if phase == "train":
            model.train()

        with tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}") as t:
            for batch in t:
                iters += 1

                ids = batch["input_ids"].to(device, dtype = torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)
                labels = batch["labels"].to(device, dtype=torch.long)

                loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=False)

                epoch_loss += loss.item()

                # Accuracyの計算
                flatted_labels = labels.view(-1)    # [batch_size * seq_len, ]
                active_logits = tr_logits.view(-1, model.num_labels)    # [batch_size * seqlen, num_labels]
                flatted_predictions = torch.argmax(active_logits, axis=1)   # [batch_size * seq_len, ]

                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)

                labels = torch.masked_select(flatted_labels, active_accuracy)
                predictions = torch.masked_select(flatted_predictions, active_accuracy)

                tr_acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                epoch_acc += tr_acc

                # 勾配クリッピング(Normで)
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config["max_grad_norm"])

                # バックプロパゲーション
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix_str(f"Loss: {loss.item():.4f}, Acc: {tr_acc:.4f}")

        torch.cuda.empty_cache()

        print(f"Epoch Loss: {epoch_loss / iters}, Epoch Acc: {epoch_acc / iters}")
        logs.append({"epoch": epoch + 1, "train_loss": epoch_loss / iters})
        df = pd.DataFrame(logs)
        df.to_csv(f"/logs/{model_path}.csv")


def main(config):

    model_checkpoint = config['model_checkpoint']
    model_name = model_checkpoint.split("/")[-1]
    model_path = f"{model_name}-{config['EXP_NUM']}"

    downloaded_model_path = config['DOWNLOADED_MODEL_PATH']

    if downloaded_model_path == "models":
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # model weightのダウンロード
        backbone = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=config['N_LABELS'])
        backbone.save_pretrained(os.path.join("models", model_name))

    # load data
    print('loading data...')
    train_df, _, lookups = read_kfold_file(config['datafile_rootpath'], config['fold'])

    print('loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    train_dataset = CustomDataset(train_df, lookups, tokenizer, config['max_length'], val=False)

    train_params = {'batch_size': config['train_batch_size'], 'shuffle': True, 'num_workers': 2, 'pin_memory':True}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_params)

    print('loading model...')
    config_model = AutoConfig.from_pretrained(downloaded_model_path + "/config.json")
    model = AutoModelForTokenClassification.from_pretrained(downloaded_model_path + "/pytorch_model.bin", config=config_model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rates"][0])

    train(model, train_dataloader, optimizer, config)

    torch.save(model.state_dict(), os.path.join('models', model_path))


if __name__ == '__main__':

    # load configuration file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='configuration file',
    )
    args = parser.parse_args()

    config_file = args.config

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    pprint(config)

    config.update({
        '__filename__': os.path.splitext(os.path.basename(config_file))[0]
    })

    main(config)