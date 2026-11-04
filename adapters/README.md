# Adapters

Registry adapters ship inside the package: `src/sphota/data/v1/adapters/` (`registry.json`
+ one `.npz` per model, SHA-256 checksummed). This folder holds recipes and docs.

* Train your own: `sphota train-adapter --model <sentence-transformers id>`
* Certify it: `sphota certify --model <id> --adapter <file.npz>`
* Proprietary APIs: [`recipes/openai_adapter.py`](recipes/openai_adapter.py),
  [`recipes/cohere_adapter.py`](recipes/cohere_adapter.py) (bring your own key; not run in CI;
  we do not distribute adapters trained on API outputs).
* Non-commercial-licensed models (e.g. jina-embeddings-v3, CC-BY-NC): train locally with
  `sphota train-adapter`; such adapters are not accepted into the default registry.

See [docs/adapters.md](../docs/adapters.md) for the measured certificates and the
unsupervised (vec2vec-style) experiment.
