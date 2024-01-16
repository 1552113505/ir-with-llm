# -*- coding: utf-8 -*-
import json
import random
from typing import *

from utils.data_utils import construct_kshots_datas


class PromptManager(object):
    def __init__(self, query: str, documents: List[str]):
        self.query = query
        self.documents = documents
        
    def generate(self):
        raise NotImplementedError 
        

class PairwisePrompt(PromptManager):
    def __init__(self, query: str, documents: List[str], examples: List[str] = None, is_k_shots: bool = False):
        super().__init__(query=query, documents=documents)

        self._examples = examples

        if not is_k_shots:
            self.prompt_template = """ If the document below aligns with the ensuing query, output ’true’, else output ’false’: query: [{}] document: [{}] relevant:"""
        else:
            self.prompt_template = """ If the document below aligns with the ensuing query, output ’true’, else output ’false’.
            {}
            query: [{}] document: [{}] relevant:"""
   
    def generate_examples(self, k: int):
        examples = []
        for example in self._examples:
            example_pos = "query: [{}], document: [{}], relevant: {}".format(example["query"], example["pos_doc"], "true")
            example_neg = "query: [{}], document: [{}], relevant: {}".format(example["query"], example["neg_doc"], "false")
            examples.extend([example_pos, example_neg])

        return examples

    def generate(self, k: int = 0):
        prompts = []

        if k == 0:
            for document in self.documents:
                prompts.append(self.prompt_template.format(document, self.query))
        else: 
            for document in self.documents:
                prompts.append(self.prompt_template.format("\n".join(self.generate_examples(k)), document, self.query))

        return prompts


if __name__== "__main__" :
    query = "cost of interior concrete flooring"
    documents = ["Video on the cost of concrete floors. Depending on the level of complexity, concrete floors can cost as little as $2 to $6 a square foot or be as expensive as $15 to $30 a square foot. Extensive surface preparation, such as grinding, crack repair, and spall repair, can add as much as $2 per square foot to the overall cost of the floor. 2  If a full resurfacing is needed, expect to tack on another $2 to $3 per square foot (for a $4 to $5 per square foot increase).", "In most cases, budget anywhere from $4 to $8 for materials. Installation usually costs another $3 to $5 per square foot, and removal and haul away of your old flooring costs about $2 per square foot.n most cases, budget anywhere from $4 to $8 for materials. Installation usually costs another $3 to $5 per square foot, and removal and haul away of your old flooring costs about $2 per square foot."]
    pairwise_prompt = PairwisePrompt(query=query, documents=documents)
    prompts = pairwise_prompt.generate()
    #print(prompts)

    pairwise_prompt = PairwisePrompt(query=query, documents=documents, is_k_shots=True)
    prompts = pairwise_prompt.generate(k=2)
    print(prompts)

    