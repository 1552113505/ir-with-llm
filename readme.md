# Information retrieval system
### Abstract
This information retrieval system consists of two parts, one is the recall module and the other is the rerank module.
### System architecture
The complete system architecture diagram is as follows:
![img.png]("https://github.com/1552113505/ir-with-llm/blob/main/image2")<br>
The modules of the entire system are as follows:
1. Input: queries, you can enter a single query or multiple queries for batch queries

2. Search: Using the Python terrier search engine, which is a Python adaptation of terrier and is implemented in Java at the bottom. Before each retrieval, the index index needs to be built offline and stored locally. Directly call the constructed index file during retrieval to retrieve the query.

3. Prompt engineering: Build corresponding prompts for the candidate list of recalls, with strategies such as point wise, list wise, etc.

4. LLM rerank: I use the large language model llama2 to rerank the candidate list for bm25 recall, and use the In-context learning capability of the large language model to construct several different prompt implementations of 0-shot, 1-shots, 2-shots, and 3-shots to rearrange the candidate list. I obtain the sorting score by calculating the probability values of true and false produced by the large language model, and complete the evaluation of the effect by pyterrier's own evaluation function.

5. The output is related documents
### Execute Command
1. Construct index:<br>
`sh bin/construct_index.sh`
2. Retreiver:<br>
`sh bin/start_retriever.sh "hello"`
3. Reranker:<br>
`sh bin/start_reranker.sh`
4. Eval:<br>
4.1.retreiver-eval: <br>`sh bin/retriever_eval.sh`<br>
4.2.reranker-eval: <br>`sh bin/reranker_eval.sh k`where k is the k in k-shots.