# -*- coding: utf-8 -*-
import time
import torch
import pandas as pd
import huggingface_hub

from typing import *
from torch.nn.functional import softmax
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from collections import defaultdict
from src.prompt.prompt_manager import PairwisePrompt, PromptManager

huggingface_hub.login(token="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")


class LlamaReranker(object):
    def __init__(self, model="/llm/llama2/llama2_7b", temperature=0.):
        self.tokenizer = LlamaTokenizer.from_pretrained(model, padding_side = "left")
        self.model = LlamaForCausalLM.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer.get_vocab().keys())
        # self.llm = LLM(model=model)
        # self.sampling_params = SamplingParams(temperature=temperature, logprobs=vocab_size, max_tokens=1)
   
    # def __call__(self, prompts : List[str]):
    #     scores = []
    #     outputs = self.llm.generate(prompts, self.sampling_params)
    #     for idx, output in enumerate(outputs):
    #         logits = output.logprobs[0]
    #         score = logits['true'] / sum([logits['true'], logits['false']])
    #         scores.append(scores)
    #     return scores

    def __call__(self, documents, prompts):
        reranker_results = []
        true_token = self.tokenizer.encode("True")[1]
        false_token = self.tokenizer.encode("False")[1]
        inputs = self.tokenizer(prompts, padding = "longest", return_tensors = "pt")
        with torch.no_grad():
            outputs = self.model.forward(**inputs)
            print(self.model.generate(**inputs))

        logits_after_softmax = outputs.logits[:, 0, (true_token, false_token)].softmax(dim=1)
        logits_list = logits_after_softmax.cpu().detach().tolist()
        print(logits_list)
        sorted_results = sorted(range(len(logits_list)), key = lambda x:logits_list[x], reverse = True)
        for sorted_result in sorted_results:
            reranker_results.append(documents[sorted_result])
        return reranker_results
        

# import pyterrier as pt
# if not pt.started():
#     pt.init()
# from pyterrier import TransformerBase

# class Wrapper(TransformerBase)
#     def __init__(prompt, model="/llm/llama2/llama2_7b", temperature=0.):
#         self.model = LlamaReranker(model=model, temperature=0.)
#         self.prompt = promot
#     def transform(self, df : pd.DataFrame):
#         prompts = df.apply(lambda x : self.prompt(x.query, x.text), axis=1)
#         scores = self.model(prompts)
#         df['score'] = scores

#         from pt.model import add_ranks
#         return add_ranks(df)
        


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
    # print(results)