# X / Twitter thread

1/ Every sentence has an address now. Sphota is GPS for meaning: any text, any language → a short
code like JYZA-7M-X0-A8. Frozen forever. Open source. 🧭 [Atlas GIF]

2/ The first block is a broad region (1 of 4096). Each next block zooms in. Same meaning in
Spanish or Japanese lands in the same region — and the code tells you how sure it is, like GPS
accuracy.

3/ Embeddings are private dialects: switch models and you re-embed everything. Sphota is a shared,
frozen space; small adapters bring models in. Index with MiniLM, query with another model:
96–99% recall@10, no re-indexing.

4/ Codes are just strings: GROUP BY them in SQL, route by prefix, cache LLM answers across
languages, diff the topics of two documents.

5/ Honest: discretizing meaning cuts some equivalent texts apart. Translations share the L1 code
46% of the time, paraphrases 72%. We publish every number and the negative results.

6/ pip install sphota · runs on CPU and in your browser · spec + paper + benchmark in the repo.
github.com/thiruvikramanpravartak/sphota
