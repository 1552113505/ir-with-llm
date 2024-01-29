# -*- coding: utf-8 -*-
import json
import time
import torch
import pandas as pd
import huggingface_hub
import pyterrier as pt
from collections import defaultdict

import transformers
from typing import *
from tqdm import tqdm
from torch.nn.functional import softmax
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from collections import defaultdict
from src.prompt.prompt_manager import PairwisePrompt, PromptManager
from pyterrier import Transformer
from pyterrier.model import add_ranks

from utils.data_utils import construct_kshots_datas

huggingface_hub.login(token="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")

if not pt.started():
    pt.init()

torch.cuda.empty_cache()


class LlamaRerankerPairwise(Transformer):
    def __init__(self, tokenizer, model, batch_size : int = 1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = 32000 # This stops nans for some reason, I trust github issues
        self.A = self.tokenizer.encode("A")[1] # lower case to suit the prompt
        self.B = self.tokenizer.encode("B")[1] # ^^
        self.batch_size = batch_size

    def split_data_by_batch(self, prompts: List[str]):
        batch_data_list = []
        for idx in range(0, len(prompts), self.batch_size):
            batch_data_list.append(prompts[idx: idx+self.batch_size])

        return batch_data_list

    def llm_predictor(self, prompts: List[str]):
        outputs = []
        with torch.no_grad():
            for prompt in prompts: # Save a few LOC just directly iterating
                toks = self.tokenizer([prompt], padding="longest", return_tensors="pt", add_special_tokens=True).to(self.model.device) # Same as before
                logits = self.model.forward(**toks).logits
                logits_after_softmax = logits[:, 0, (self.A, self.B)].log_softmax(dim=1) # Log softmax is generally used, can't remember why
                outputs.extend(logits_after_softmax.cpu().detach().tolist())

        return outputs

    def rerank(self, query: str, documents: List[str]):
        scores = {}
        for i in range(len(documents)):
            for j in range(i, len(documents)):
                i_j_prompt = PairwisePrompt(query=query, passage_a=documents[i], passage_b=documents[j]).generate()[0]
                j_i_prompt = PairwisePrompt(query=query, passage_a=documents[j], passage_b=documents[i]).generate()[0]
                llm_predictor_results = self.llm_predictor([i_j_prompt, j_i_prompt])

                if llm_predictor_results[0][0] > llm_predictor_results[0][1]:
                    scores[documents[i]] = scores.get(documents[i], 0) + 1
                elif llm_predictor_results[0][0] < llm_predictor_results[0][1]:
                    scores[documents[j]] = scores.get(documents[j], 0) + 1
                else:
                    scores[documents[i]] = scores.get(documents[i], 0) + 0.5
        
        return list(scores.values())
                
    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        scores = []
        q2d = defaultdict(list)
        for _, row in topics_or_res.iterrows():
            q2d[row["query"]].append(row["text"])
            
        for q, ds in tqdm(q2d.items()):
            score = self.rerank(q, ds)
            scores.extend(score)
            
        topics_or_res["score"] = scores
        
        return add_ranks(topics_or_res)


class LlamaRerankerPairwiseKShots(LlamaRerankerPairwise):
    def __init__(self, tokenizer, model, k: int, batch_size : int = 1):
        super().__init__(tokenizer, model, batch_size)
        self._k= k
        self._examples = self.load_examples(k)

    def load_examples(self, k: int):
        examples = []
        with open(f"data/k_shots/{k}_shots_examples.json") as fi:
            for line in fi:
                line = line.strip()
                examples.append(json.loads(line))

        return examples

    def rerank(self, query: str, documents: List[str]):
        scores = {}
        for i in range(len(documents)):
            for j in range(i, len(documents)):
                i_j_prompt = PairwisePrompt(query=query, passage_a=documents[i], passage_b=documents[j], examples=self._examples, is_k_shots=True).generate(self._k)[0]
                j_i_prompt = PairwisePrompt(query=query, passage_a=documents[j], passage_b=documents[i], examples=self._examples, is_k_shots=True).generate(self._k)[0]
                llm_predictor_results = self.llm_predictor([i_j_prompt, j_i_prompt])

                if llm_predictor_results[0][0] > llm_predictor_results[0][1]:
                    scores[documents[i]] = scores.get(documents[i], 0) + 1
                elif llm_predictor_results[0][0] < llm_predictor_results[0][1]:
                    scores[documents[j]] = scores.get(documents[j], 0) + 1
                else:
                    scores[documents[i]] = scores.get(documents[i], 0) + 0.5
        
        return list(scores.values())
    
    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        scores = []
        q2d = defaultdict(list)
        for _, row in topics_or_res.iterrows():
            q2d[row["query"]].append(row["text"])
            
        for q, ds in tqdm(q2d.items()):
            score = self.rerank(q, ds)
            scores.extend(score)
            
        topics_or_res["score"] = scores
        
        return add_ranks(topics_or_res)


if __name__== "__main__" :
    query = "cost of interior concrete flooring"
    # documents = ["Time: 02:53. Video on the cost of concrete floors. The cost of a concrete floor is economical, about $2 to $6 per square foot depending on the level of complexity.", "Size of the Floor - Typically, the larger the floor area, the lower the cost per square foot for installation due to the economies of scale. A small residential floor project, for example, is likely to cost more per square foot than a large 50,000-square-foot commercial floor."]
    documents = ["Time: 02:53. Video on the cost of concrete floors. The cost of a concrete floor is economical, about $2 to $6 per square foot depending on the level of complexity.", "Polished Concrete Prices. Polished concrete pricing and cost per square metre for concrete polishing can range from as little as Â£20 to as much as Â£150 per metre square. Additionally polished concrete overlays with exotic aggregates could add further costs of Â£10 to Â£50 per metre.olished concrete pricing and cost per square metre for concrete polishing can range from as little as Â£20 to as much as Â£150 per metre square.", "Stained Concrete Cost. With staining, itâs often possible to dress up plain gray concrete for less than the cost of covering it up with carpeting, tile, or most types of high-end flooring materials. At this price point, you are comparing against wood flooring ($8-$10 per square foot) and a range of ceramic and quarry tiles ($10-$12 per square foot). 2  Adding decorative sandblasting or engraving to the advanced stain application ($15+ per square foot)."]
    # documents = [documents[0]]
    pairwise_prompt = PairwisePrompt(query=query, documents=documents)
    prompts = pairwise_prompt.generate()
    #print(prompts)

    documents = ["Time: 02:53. Video on the cost of concrete floors. The cost of a concrete floor is economical, about $2 to $6 per square foot depending on the level of complexity.", "Polished Concrete Prices. Polished concrete pricing and cost per square metre for concrete polishing can range from as little as Â£20 to as much as Â£150 per metre square. Additionally polished concrete overlays with exotic aggregates could add further costs of Â£10 to Â£50 per metre.olished concrete pricing and cost per square metre for concrete polishing can range from as little as Â£20 to as much as Â£150 per metre square.", "Stained Concrete Cost. With staining, itâs often possible to dress up plain gray concrete for less than the cost of covering it up with carpeting, tile, or most types of high-end flooring materials. At this price point, you are comparing against wood flooring ($8-$10 per square foot) and a range of ceramic and quarry tiles ($10-$12 per square foot). 2  Adding decorative sandblasting or engraving to the advanced stain application ($15+ per square foot)."]
    frame = pd.DataFrame({'docno' : [*range(len(documents))], 'text': documents})
    frame['qid'] = 1
    frame['query'] = "cost of interior concrete flooring"
    tokenizer = LlamaTokenizer.from_pretrained('/llm/llama2/llama2_7b', padding_side = "left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llama_reranker = LlamaReranker(model=LlamaForCausalLM.from_pretrained('/llm/llama2/llama2_7b', device_map="auto"), 
                                    tokenizer=tokenizer)
    results = llama_reranker(frame)
    print(results)