# -*- coding: utf-8 -*-
import configparser
import pyterrier as pt
from ir_measures import nDCG, AP

from utils.data_utils import construct_topics, construct_qrels


if not pt.started():
    pt.init()


def evaluation(conf_file = None):
    """
    评估
    :conf_file:
    :return:
    """
    topic_file_path = None
    qrels_file_path = None
    index_file_path = None
    if conf_file is not None:
        config = configparser.ConfigParser()
        config.read(conf_file, encoding="utf-8")
        topic_file_path = config["eval"]["topic_file_path"]
        qrels_file_path = config["eval"]["qrels_file_path"]
        index_file_path = config["eval"]["location_path"]

    topics = construct_topics(topic_file_path)
    qrels = construct_qrels(qrels_file_path)

    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25")

    result = pt.Experiment(
        [bm25],
        topics,
        qrels,
        names=["BM25"],
        eval_metrics=[nDCG@10, AP(rel=2)]
    )

    return result


if __name__ == '__main__':
    conf_file = "./conf/system.conf"
    result = evaluation()
    print(result)
