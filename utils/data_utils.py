# -*- coding: utf-8 -*-
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
