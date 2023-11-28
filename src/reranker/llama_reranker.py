# -*- coding: utf-8 -*-
import time
import torch
import pandas as pd
import huggingface_hub 

from typing import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.prompt.prompt_manager import PairwisePrompt, PromptManager

huggingface_hub.login(token="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")


class LlamaReranker(object):
    def __init__(self, model="/llm/llama2/llama2_7b", temperature=0.8, top_p=0.95):
        self.llm = LLM(model=model)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    def __call__(self, prompts: List[str]):
        results = []
        outputs = self.llm.generate(prompts)
        for output in outputs:
            results.append(output.outputs[0].text)

        return results


if __name__== "__main__" :
    query = "cost of interior concrete flooring"
    documents = ["Video on the cost of concrete floors. Depending on the level of complexity, concrete floors can cost as little as $2 to $6 a square foot or be as expensive as $15 to $30 a square foot. Extensive surface preparation, such as grinding, crack repair, and spall repair, can add as much as $2 per square foot to the overall cost of the floor. 2  If a full resurfacing is needed, expect to tack on another $2 to $3 per square foot (for a $4 to $5 per square foot increase).", "In most cases, budget anywhere from $4 to $8 for materials. Installation usually costs another $3 to $5 per square foot, and removal and haul away of your old flooring costs about $2 per square foot.n most cases, budget anywhere from $4 to $8 for materials. Installation usually costs another $3 to $5 per square foot, and removal and haul away of your old flooring costs about $2 per square foot."]
    pairwise_prompt = PairwisePrompt(query=query, documents=documents)
    prompts = pairwise_prompt.generate()
    
    llama_reranker = LlamaReranker()
    results = llama_reranker(prompts)
    print(results)