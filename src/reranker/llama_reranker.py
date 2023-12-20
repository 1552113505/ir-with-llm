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
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from collections import defaultdict
from src.prompt.prompt_manager import PairwisePrompt, PromptManager
from pyterrier import Transformer
from pyterrier.model import add_ranks

huggingface_hub.login(token="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")
if not pt.started():
    pt.init()


use_8bit = True
bnb_8bit_compute_dtype = "float16"
bnb_8bit_quant_type = "nf8"
compute_dtype = getattr(torch, bnb_8bit_compute_dtype)
use_nested_quant = False

bnb_config = BitsAndBytesConfig(
    load_in_8bit=use_8bit,
    bnb_8bit_quant_type=bnb_8bit_quant_type,
    bnb_8bit_compute_dtype=compute_dtype,
    bnb_8bit_use_double_quant=use_nested_quant
)
#torch.cuda.clear_cache()
torch.cuda.empty_cache()


class LlamaReranker(Transformer):
    def __init__(self, model, tokenizer, batch_size : int = 1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = 32000 # This stops nans for some reason, I trust github issues
        self.TRUE = self.tokenizer.encode("true")[1] # lower case to suit the prompt
        self.FALSE = self.tokenizer.encode("false")[1] # ^^
        self.batch_size = batch_size

    def split_data_by_batch(self, prompts: List[str]):
        batch_data_list = []
        for idx in range(0, len(prompts), self.batch_size):
            batch_data_list.append(prompts[idx: idx+self.batch_size])

        return batch_data_list

    def score(self, prompts: List[str], documents: List[str] = None):
        print("start model predict...")
        outputs = []
        with torch.no_grad():
            for batch in tqdm(self.split_data_by_batch(prompts)): # Save a few LOC just directly iterating
                toks = self.tokenizer(batch, padding="longest", return_tensors="pt", add_special_tokens=True).to(self.model.device) # Same as before
                logits = self.model.forward(**toks).logits
                logits_after_softmax = logits[:, 0, (self.TRUE, self.FALSE)].log_softmax(dim=1) # Log softmax is generally used, can't remember why
                print(logits_after_softmax[:, 0].cpu().detach().tolist())
                outputs.extend(logits_after_softmax[:, 0].cpu().detach().tolist()) # Get just the softmax prob of 'true'

        return outputs
    
    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        prompts = topics_or_res.apply(lambda x : PairwisePrompt(query=x.query, documents=[x.text]).generate()[0], axis=1) # Point wise prompt creation
        prompts = list(prompts.to_dict().values())
        scores = self.score(prompts)
        topics_or_res["score"] = scores # ordered input therefore can assign list directly
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
    llama_reranker = LlamaReranker(model=LlamaForCausalLM.from_pretrained('/llm/llama2/llama2_7b', quantization_config=bnb_config, device_map="auto"), 
                                    tokenizer=tokenizer)
    results = llama_reranker(frame)
    # print("\n".join(results))
    print(results)