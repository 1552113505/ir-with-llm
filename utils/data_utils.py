# -*- coding: utf-8 -*-
import pandas as pd
import pyterrier as pt
from typing import Dict
from tqdm import tqdm


def transfer_tsv_id2doc(data_file: str) -> Dict[str, str]:
    """
    将tsv文件转换成id2doc的字典
    :param data_file: tsv文件地址
    :return: 字典
    """
    id2doc = {}
    with pt.io.autoopen(data_file, "rt") as file:
        for line in tqdm(file):
            docno, doc = line.split("\t")
            id2doc[docno] = doc

    return id2doc


def remove_punctuation_v2(text):
    """删除标点符号"""
    for char in text:
        if char in [".", ",", ":", ";", "?", "!", "'", "/"]:
            text = text.replace(char, '')
    return text


def construct_topics(topic_file_path):
    """
    构造查询集
    :param topic_file_path: 查询集地址
    :return: pd.DataFrame["qid", "query"]
    """
    ids, datas = [], []
    with pt.io.autoopen(topic_file_path, "rt") as file:
        for idx, line in tqdm(enumerate(file)):
            qid, query = line.strip().split("\t")
            query = remove_punctuation_v2(query)
            ids.append(idx)
            datas.append([qid, query])

    topics = pd.DataFrame(
        data=datas,
        index=ids,
        columns=["qid", "query"]
    )

    return topics


def construct_qrels(qrels_file_path):
    """
    构造qrels数据集
    :param qrels_file_path: qrels文件地址
    :return: pd.DataFrame["qid", "docno", "label"]
    """
    ids, datas = [], []
    with pt.io.autoopen(qrels_file_path, "rt") as file:
        for idx, line in tqdm(enumerate(file)):
            qid, _, doc_id, label = line.strip().split()
            ids.append(idx)
            datas.append([qid, doc_id, int(label)])

    qrels = pd.DataFrame(
        data=datas,
        index=ids,
        columns=["qid", "docno", "label"]
    )

    return qrels
