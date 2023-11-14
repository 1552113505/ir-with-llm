# -*- coding: utf-8 -*-
import configparser
import sys

import pyterrier as pt
import pandas as pd
from pprint import pprint
from typing import List

from utils.data_utils import transfer_tsv_id2doc

pt.init()


class BM25Retriever(object):
    """
    召回期
    """
    def __init__(self, conf_file: str):
        self._config = configparser.ConfigParser()
        self._config.read(conf_file, encoding="utf-8")
        self._index_location = self._config["retriever"]["location_path"]
        self._dataset_path = self._config["retriever"]["data_file"]

    def __call__(self, queries: List[str], num_results=10, is_recall_doc=True):
        """
        召回
        :param queries:
        :return:
        """
        bm25 = pt.BatchRetrieve(index_location=self._index_location,
                                meta={'docno': 20, 'text': 4096},
                                wmodle="bm25",
                                num_results=num_results)
        recall_list = bm25.transform(queries)
        if is_recall_doc:
            documents = []
            id2doc = transfer_tsv_id2doc(self._dataset_path)
            doc_ids = recall_list.docid
            for doc_id in doc_ids:
                documents.append(id2doc[str(doc_id)])
            # recall_list["document"] = documents

        return documents


if __name__ == '__main__':
    query = sys.argv[1]
    conf_file = ".\\conf\\system.conf"
    bm25_retrevier = BM25Retriever(conf_file=conf_file)
    recall_list = bm25_retrevier([query], is_recall_doc=True)
    pprint(recall_list)
