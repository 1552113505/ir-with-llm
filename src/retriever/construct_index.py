# -*- coding: utf-8 -*-
import traceback
import configparser
import pyterrier as pt
from typing import Dict
from tqdm import tqdm

pt.init()


class ConstructIndex(object):
    """
    离线构建索引
    """

    def __init__(self, conf_file: str):
        self._config = configparser.ConfigParser()
        self._config.read(conf_file, encoding="utf-8")
        self._dataset_path = self._config["construct_index"]["data_file"]
        self._index_location = self._config["construct_index"]["location_path"]

    def _load_dataset(self) -> Dict[str, str]:
        """
        加载数据集，这里主要是trec_deep_learning的tsv文件
        :return: 构建好的数据词典
        """
        with pt.io.autoopen(self._dataset_path, "rt") as file: # 安全的读文件
            for line in tqdm(file): # 进度条
                docno, doc = line.split("\t")
                yield {"docno": docno, "text": doc}

    def construct(self, wmodel: str = "bm25") -> bool:
        """
        构建索引
        :return: 构建成功还是失败
        """
        construct_flag = False
        try:
            iter_indexer = pt.IterDictIndexer(self._index_location,
                                              meta={"docno": 20, "text": 4096},
                                              # stopwords=pt.TerrierStopwords.terrier,
                                              # stemmer="porter",
                                              wmodel=wmodel)
            # indexref = iter_indexer.index(self._load_dataset(), meta=["docno", "text"])
            indexref = iter_indexer.index(self._load_dataset())
            construct_flag = True
        except Exception as e:
            print(f"construct index failed, reason: {traceback.format_exc()}")

        return construct_flag


if __name__ == '__main__':
    conf_file = "./conf/system.conf"
    indexer = ConstructIndex(conf_file=conf_file)
    flag = indexer.construct()
    print(flag)
