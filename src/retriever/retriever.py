# -*- coding: utf-8 -*-
import pyterrier as pt
import pandas as pd
from pprint import pprint
from typing import List

from utils.data_utils import transfer_tsv_id2doc

pt.init()
pd.set_option('display.max_columns', None)


class BM25Retriever(object):
    """
    召回期
    """
    def __init__(self, index_location: str, dataset_path: str):
        self._index_location = index_location
        self._dataset_path = dataset_path

    def __call__(self, queries: List[str], num_results=10, is_recall_doc=True):
        """
        召回
        :param queries:
        :return:
        """
        bm25 = pt.BatchRetrieve(index_location=self._index_location,
                                meta={'docno': 20, 'text': 4096},
                                # metadata=["text"],
                                wmodle="bm25",
                                num_results=num_results)
        recall_list = bm25.transform(queries)
        if is_recall_doc:
            docments = []
            id2doc = transfer_tsv_id2doc(self._dataset_path)
            doc_ids = recall_list.docid
            for doc_id in doc_ids:
                docments.append(id2doc[str(doc_id)])
            recall_list["document"] = docments

        return recall_list


if __name__ == '__main__':
    bm25_retrevier = BM25Retriever(index_location="./data/index/msmacro_stemmer_bm25",
                                   dataset_path="./data/corpus/collection.tsv")
    recall_list = bm25_retrevier(["the cost of a postal money"], is_recall_doc=True)
    pprint(recall_list.to_dict())
    # pprint(recall_list)


