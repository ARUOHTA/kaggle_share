import os, sys, tqdm
from typing import Tuple, Dict
import numpy as np
import pandas as pd


def score_feedback_comp(pred_df: pd.DataFrame, ner_df:pd.DataFrame) -> Tuple[int, Dict[str, int]]:
    """
    Feedback Prizeのデフォルトの提出csvと読み込んだcsvから各クラスのF1スコアを算出する

    Parameter
    ---------
    pred_df: pd.Dataframe
        column = [["id", "class", "predictionstring"]]
    ner_df: pd.Dataframe
        column = [["id", "text", "classlist", "predictionstrings", "annotation"]]
    
    Return
    ------
    int:
        全体のF1スコア
    Dict[str, int]:
        keyクラスのF1スコア
    """
    # ground_truth dataframeの作成
    gt_df = ner_df[["id", "classlist", "predictionstrings"]].reset_index(drop=True).copy()
    gt_df["predictionstring"] = gt_df["predictionstrings"]
    gt_df = gt_df[["id", "classlist", "predictionstring"]].reset_index(drop=True).copy()
    gt_df = gt_df.explode(["classlist", "predictionstring"]).reset_index(drop=True)

    f1s = {}
    CLASSES = pred_df["class"].unique()
    for c in tqdm.tqdm(CLASSES):
        p_df = pred_df.loc[pred_df["class"] == c].copy()
        g_df = gt_df.loc[gt_df["classlist"] == c].copy()
        f1 = _calc_f1_score(p_df, g_df)
        f1s[c] = f1
    f1 = np.mean([i for i in f1s.values()])
    return f1, f1s




def _calc_f1_score(pred_df: pd.DataFrame, gt_df:pd.DataFrame) -> int:
    """
    Feedback Prizeのデフォルトの提出csvと読み込んだcsvからF1スコアを算出する

    Parameter
    ---------
    pred_df: pd.Dataframe
        column = [["id", "class", "predictionstring"]]
    ner_df: pd.Dataframe
        column = [["id", "classlist", "predictionstring"]]
    """
    """# ground_truth dataframeの作成
    gt_df = ner_df[["id", "classlist", "predictionstrings"]].reset_index(drop=True).copy()
    gt_df["predictionstring"] = gt_df["predictionstrings"]
    gt_df = gt_df[["id", "classlist", "predictionstring"]].reset_index(drop=True).copy()
    gt_df = gt_df.explode(["classlist", "predictionstring"]).reset_index(drop=True)"""
    gt_df["gt_id"] = gt_df.index

    # pred_dfの準備
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index


    # 1. 全ての正解ラベルと予測を比べる
    joined = pred_df.merge(gt_df, left_on=["id", "class"], right_on=["id", "classlist"], how="outer", suffixes=("_pred", "_gt"))
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")
    joined["overlaps"] = joined.apply(_calc_overlap, axis=1)


    # 2. 正解と予測のオーバラップが>=0.5でかつ予測と正解のオーバーラップが>=0.5の時TPとする
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])
    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)

    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)

    tp_pred_ids = joined.query("potential_TP").sort_values("max_overlap", ascending=False).groupby(["id", "predictionstring_gt"]).first()["pred_id"].values


    # 3, 正解ラベルと一致する予測がない時にFN, 予測と一致する正解ラベルがない時にFPとする
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    
    # 4. microF1スコアを計算
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    mi_f1_score = TP / (TP + 0.5 * (FP + FN))

    return mi_f1_score



def _calc_overlap(row):
    # 集合に変換
    set_pred = set(row.predictionstring_pred.split(" "))
    set_groundtruth = set(row.predictionstring_gt.split(" "))

    len_pred = len(set_pred)
    len_groundtruth = len(set_groundtruth)

    # 被りを計算し返す
    inter = len(set_groundtruth.intersection(set_pred))
    overlap1 = inter / len_groundtruth
    overlap2 = inter / len_pred
    return [overlap1, overlap2]