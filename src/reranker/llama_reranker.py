# -*- coding: utf-8 -*-
import time
import torch
import pandas as pd
import huggingface_hub

from typing import *
from vllm import LLM, SamplingParams
from collections import defaultdict
from src.prompt.prompt_manager import PairwisePrompt, PromptManager

huggingface_hub.login(token="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")


class LlamaReranker(object):
    def __init__(self, model="/llm/llama2/llama2_7b", temperature=0.8, top_p=0.95):
        self.llm = LLM(model=model)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    def __call__(self, documents: List[str], prompts: List[str]):
        reranker_results = []
        true_text_dict = {}
        false_text_dict = {}

        outputs = self.llm.generate(prompts, self.sampling_params)
        for idx, output in enumerate(outputs):
            generate_text = output.outputs[0].text
            print(generate_text)
            generate_prob = output.outputs[0].cumulative_logprob
            if "True" in generate_text or "true" in generate_text:
              true_text_dict[idx] = generate_prob
            else:
              false_text_dict[idx] = generate_prob
        print(true_text_dict)
        true_results_sorted = sorted(true_text_dict.items(), key=lambda x: x[1], reverse=True)
        false_results_sorted = sorted(false_text_dict.items(), key=lambda x: x[1], reverse=False)

        for item in true_results_sorted:
          reranker_results.append(documents[item[0]])
        for item in false_results_sorted:
          reranker_results.append(documents[item[0]])

        return reranker_results


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