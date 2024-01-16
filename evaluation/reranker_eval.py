# -*- coding: utf-8 -*-
import sys
import torch
import ir_datasets
import configparser
import pyterrier as pt
from ir_measures import nDCG, AP
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from utils.data_utils import construct_topics, construct_qrels
from src.reranker.llama_reranker import LlamaReranker, LlamaRerankerKShots

if not pt.started():
    pt.init()

torch.cuda.empty_cache()


def evaluation(k = 0, conf_file = None):
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

    tokenizer = LlamaTokenizer.from_pretrained('/llm/llama2/llama2_7b', padding_side = "left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = LlamaForCausalLM.from_pretrained('/llm/llama2/llama2_7b', torch_dtype=torch.float16, device_map={"": 0})

    if k == 0:
        llama_reranker = LlamaReranker(model=model, 
                                        tokenizer=tokenizer)
    else:
        llama_reranker = LlamaRerankerKShots(model=model, 
                                        tokenizer=tokenizer,
                                        k=k)

    text_ref = pt.get_dataset('irds:msmarco-passage')
    bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed", wmodel="BM25")
    pipe = (bm25 % 100) >> pt.text.get_text(text_ref, 'text') >> llama_reranker
    result = pt.Experiment(
        [pipe],
        topics,
        qrels,
        names=["BM25"],
        eval_metrics=[nDCG@10, AP(rel=2)]
    )

    return result


if __name__ == '__main__':
    k = int(sys.argv[1])
    conf_file = "./conf/system.conf"
    result = evaluation(k=k)
    print(result)