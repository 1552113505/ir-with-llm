# -*- coding: utf-8 -*-
import time
import torch
import pandas as pd
import huggingface_hub
import pyterrier as pt

from typing import *
from tqdm import tqdm
from torch.nn.functional import softmax
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from collections import defaultdict
from src.prompt.prompt_manager import PairwisePrompt, PromptManager
from pyterrier import Transformer
from pyterrier.model import add_ranks

huggingface_hub.login(token="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")
if not pt.started():
    pt.init()


class LlamaReranker(object):
    def __init__(self, model="/llm/llama2/llama2_7b", temperature=0.):
        self.batch_size = 8
        self.tokenizer = LlamaTokenizer.from_pretrained(model, padding_side = "left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer.get_vocab().keys())
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.device = "cpu"
        self.model = LlamaForCausalLM.from_pretrained(model).to(self.device)
        # self.llm = LLM(model=model)
        # self.sampling_params = SamplingParams(temperature=temperature, logprobs=vocab_size)

    def split_data_by_batch(self, prompts: List[str]):
        batch_data_list = []
        for idx in range(0, len(prompts), self.batch_size):
            batch_data_list.append(prompts[idx: idx+self.batch_size])

        return batch_data_list
        
    def __call__(self, prompts: List[str], documents: List[str] = None):
        reranker_results = []

        true_token = self.tokenizer.encode("True")[1]
        false_token = self.tokenizer.encode("False")[1]

        batch_data_list = self.split_data_by_batch(prompts)

        print("start tokenizer...")
        inputs = []
        for batch_data in tqdm(batch_data_list):
            inputs.append(self.tokenizer(batch_data, padding="longest", return_tensors="pt").to(self.device))

        print("start model predict...")
        outputs = []
        with torch.no_grad():
            for input in tqdm(inputs):
                outputs.extend(self.model.forward(**input))

        logits_after_softmax = outputs.logits[:, 0, (true_token, false_token)].softmax(dim=1)
        logits_list = logits_after_softmax.cpu().detach().tolist()

        scores = [item[0] for item in logits_list]
        sorted_results = sorted(range(len(logits_list)), key=lambda x: logits_list[x][0], reverse=True)

        if documents is not None:
            for sorted_result in sorted_results:
                reranker_results.append(documents[sorted_result])

        return scores, reranker_results
        

class LlamaRerankerWrapper(Transformer):
    def __init__(self, llama_reranker: LlamaReranker):
        super().__init__()
        self._llama_reranker = llama_reranker
    
    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        prompts = topics_or_res.apply(lambda x: PairwisePrompt(query=x.query, documents=[x.text]).generate()[0], axis=1)
        # print(prompts.to_dict())
        prompts = list(prompts.to_dict().values())
        scores, _ = self._llama_reranker(prompts)
        topics_or_res["score"] = scores

        return add_ranks(topics_or_res)
      

if __name__== "__main__" :
    query = "cost of interior concrete flooring"
    # documents = ["Time: 02:53. Video on the cost of concrete floors. The cost of a concrete floor is economical, about $2 to $6 per square foot depending on the level of complexity.", "Size of the Floor - Typically, the larger the floor area, the lower the cost per square foot for installation due to the economies of scale. A small residential floor project, for example, is likely to cost more per square foot than a large 50,000-square-foot commercial floor."]
    documents = ["Time: 02:53. Video on the cost of concrete floors. The cost of a concrete floor is economical, about $2 to $6 per square foot depending on the level of complexity.", "Polished Concrete Prices. Polished concrete pricing and cost per square metre for concrete polishing can range from as little as Â£20 to as much as Â£150 per metre square. Additionally polished concrete overlays with exotic aggregates could add further costs of Â£10 to Â£50 per metre.olished concrete pricing and cost per square metre for concrete polishing can range from as little as Â£20 to as much as Â£150 per metre square.", "Stained Concrete Cost. With staining, itâs often possible to dress up plain gray concrete for less than the cost of covering it up with carpeting, tile, or most types of high-end flooring materials. At this price point, you are comparing against wood flooring ($8-$10 per square foot) and a range of ceramic and quarry tiles ($10-$12 per square foot). 2  Adding decorative sandblasting or engraving to the advanced stain application ($15+ per square foot)."]
    # documents = [documents[0]]
    pairwise_prompt = PairwisePrompt(query=query, documents=documents)
    prompts = pairwise_prompt.generate()
    print(prompts)

    llama_reranker = LlamaReranker()
    results = llama_reranker(documents, prompts)
    # print("\n".join(results))
    print(results)