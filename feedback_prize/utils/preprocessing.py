import os, sys, tqdm, ast, pathlib
from random import random
from typing import Tuple
import pandas as pd
from sklearn.model_selection import KFold



def make_NER_dataframe(datafile_rootpath: str, save_data=True, n_splits: int or bool = 5, seed=1234):
    """
    Feedback Prizeのデフォルトのファイル構造から、IDとdocumentとNERラベルをまとめたファイルをバリデーションを行ってdetaframeを返す関数

    Parameter
    ---------
    datafile_rootpath: Feedback Prize Datasetのデフォルトのファイル構造を保存したパス(ex.: ./data/test, ./data/train, ./data/train.csv -> "./data")
    save_data: 作ったデータフレームを`datafile_rootpath`下に保存するかどうか
    n_splits: `False`の時、CVを行わない。`int`の時、どれだけの数のfoldを作成するか。
    """
    # ラベルデータの読み込み
    train = pd.read_csv(os.path.join(datafile_rootpath, "train.csv"))
    
    path = pathlib.Path(os.path.join(datafile_rootpath, 'train'))

    def get_raw_text(ids):
        with open(path/f'{ids}.txt', 'r') as file: data = file.read()
        return data
    
    df1 = train.groupby('id')['discourse_type'].apply(list).reset_index(name='classlist')
    df2 = train.groupby('id')['predictionstring'].apply(list).reset_index(name='predictionstrings')

    df = pd.merge(df1, df2, how='inner', on='id')
    df['text'] = df['id'].apply(get_raw_text)

    # テキストをNERラベルに置き換える
    all_data = []
    with tqdm.tqdm(df.iterrows(), total=len(df)) as t:
        for k, items in enumerate(t):
            # 1. NERラベルを格納するために、テキスト長のダミーラベルを作る
            doc_length = items[1]["text"].split().__len__()
            ner_labels = ["O"] * doc_length

            # 2. 同じidのラベルデータを読み込む
            for j in train[train["id"] == items[1]["id"]].iterrows():
                # discourse typeとその範囲を読み込む
                discourse = j[1]["discourse_type"]
                list_idx = [int(x) for x in j[1]["predictionstring"].split(" ")]    

                # 開始タグをつける
                ner_labels[list_idx[0]] = f"B-{discourse}"
                # その後のNERラベル
                for l in list_idx[1:]:
                    ner_labels[l] = f"I-{discourse}"
            all_data.append(ner_labels)
    df["annotation"] = all_data

    # validationを行う
    if n_splits:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold = 0
        for train_idx, val_idx in cv.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            if save_data:
                os.makedirs(os.path.join(datafile_rootpath, str(n_splits) + "_fold", "fold" + str(fold)), exist_ok=True)
                train_df.to_csv(os.path.join(datafile_rootpath, str(n_splits) + "_fold", "fold" + str(fold), "train_ner.csv"), index=False)
                val_df.to_csv(os.path.join(datafile_rootpath, str(n_splits) + "_fold", "fold" + str(fold), "val_ner.csv"), index=False)
            fold += 1
    else:
        if save_data:
            df.to_csv(os.path.join(datafile_rootpath, "train_ner.csv"), index=False)




def read_kfold_file(datafile_rootpath: str, fold: int, n_splits: int or bool = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    make_NER_dataframeで作られたファイル構造から、指定のK-Foldの指定のfoldを読み込む関数

    Parameter
    ---------
    datafile_rootpath: Feedback Prize Datasetのデフォルトのファイル構造を保存したパス(ex.: ./data/test, ./data/train, ./data/train.csv -> "./data")
    fold: 指定の`fold`番目のfoldを読み込む(`for fold in range(n_splits)`で回せるため)
    n_splits: `False`の時、CVを行わない。`int`の時、指定されたK_Foldのファイルを読み込む

    Return
    ------
    train_df, val_df: 訓練, 検証セットのinputとtarget
    lookup_dict: NERラベルに存在するラベル名。`labels_to_ids = {v:k for k, v in enumerate(lookup_dict)}`, `ids_to_labels = {k:v for k, v in enumerate(lookup_dict)}`でencode, decode可能
    """
    # assert
    assert fold < n_splits, "fold less than n_splits"

    # データの読み込み
    train_df = pd.read_csv(os.path.join(datafile_rootpath, str(n_splits) + "_fold", "fold" + str(fold), "train_ner.csv"))
    val_df = pd.read_csv(os.path.join(datafile_rootpath, str(n_splits) + "_fold", "fold" + str(fold), "val_ner.csv"))

    train_df[["classlist", "predictionstrings", "annotation"]] = train_df[["classlist", "predictionstrings", "annotation"]].applymap(lambda x: ast.literal_eval(x))
    val_df[["classlist", "predictionstrings", "annotation"]] = val_df[["classlist", "predictionstrings", "annotation"]].applymap(lambda x: ast.literal_eval(x))

    # lookupテーブルを作る
    lookup_dict = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

    return train_df, val_df, lookup_dict




if __name__ == "__main__":
    train_df, val_df, lookups = read_kfold_file("./data", fold=1)
    print(train_df.shape)
    print(val_df.shape)
    