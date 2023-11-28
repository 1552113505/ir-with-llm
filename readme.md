# Information retrieval system
### Abstract
This information retrieval system consists of two parts, one is the recall module and the other is the rearrangement module.
### System architecture
The complete system architecture diagram is as follows:
![img.png]("https://github.com/1552113505/ir-with-llm/blob/main/image2")<br>
The modules of the entire system are as follows:
1. Input: queries, you can enter a single query or multiple queries for batch queries

2. Search: Using the Python terrier search engine, which is a Python adaptation of terrier and is implemented in Java at the bottom. Before each retrieval, the index index needs to be built offline and stored locally. Directly call the constructed index file during retrieval to retrieve the query.

3. Prompt engineering: Build corresponding prompts for the candidate list of recalls, with strategies such as point wise, list wise, etc.

4. LLM rerank: Using a large model to implement rerank work on the constructed prompt, common open source large models include Alpaca, llama, llama2, T5, and chatglm2.

5. The output is related documents
### Execute Command
1. Construct index:<br>
`python -m src.retriever.construct_index`
2. Retreiver:<br>
`python -m src.retriever.retriever "hello"`
3. Rerank:<br>
4. Eval:<br>
4.1.retreiver-eval: <br>`python -m evaluation.retriever_eval`
