# Third-party licenses

Sphota's code is Apache-2.0. The v1 release artifacts (projection matrices, codebooks,
labels, adapters) are derived from the models and datasets below. Licenses were checked
against the Hugging Face Hub metadata and model/dataset cards on 2026-11-04. Re-check them
before any public release (`scripts/license_audit.py` automates the Hub lookup).

## Models

| Model | License | Role in Sphota | Notes |
|---|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | **v1 anchor** and default encoder (Python ONNX + browser) | ONNX export at `Xenova/multilingual-e5-small` (same weights) |
| `BAAI/bge-m3` | MIT | bake-off candidate; adapter | 568M params, too slow as a CPU default |
| `sentence-transformers/LaBSE` | Apache-2.0 | bake-off candidate; adapter | strong translation alignment, weaker topical structure |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | bake-off candidate; adapter | |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | adapter (English-only) | |
| `BAAI/bge-small-en-v1.5` | MIT | adapter (English-only) | |
| `thenlper/gte-small` | MIT | adapter (English-only) | |
| `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | optional adapter recipe | not trained in the CPU build |
| `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | offline region naming (labels) | not shipped; only generated label strings |
| `Helsinki-NLP/opus-mt-en-{hi,dra,inc,es,fr,zh,ar,ru,jap,ROMANCE}` | Apache-2.0 | offline label translation | not shipped |
| `Helsinki-NLP/opus-mt-en-de` | CC-BY-4.0 | offline label translation (German) | attribution: Helsinki-NLP / OPUS-MT |
| `Xenova/*` ONNX exports | inherit source model license | ONNX encoders for registry adapters | |
| `jinaai/jina-embeddings-v3` | **CC-BY-NC-4.0** | **not used** | non-commercial weights; only a user-trained recipe is allowed |
| OpenAI / Cohere embedding APIs | proprietary ToS | recipe only (user's own key) | Sphota ships no adapter weights trained on API outputs |

Adapter weights are small linear maps (`.npz`) fitted on the outputs of the listed models.
They contain no model weights. We release them under Apache-2.0; the upstream model's own
license still governs your use of that model.

## Datasets

| Dataset | License | Used for |
|---|---|---|
| FLORES-200 (`mteb/flores`, mirror of `facebook/flores`) | CC-BY-SA-4.0 | cross-lingual alignment (dev split) and benchmark (devtest split) |
| SIB-200 (`Davlan/sib200`) | CC-BY-SA-4.0 | topic labels for evaluation only |
| Tatoeba (`mteb/tatoeba-bitext-mining`) | CC-BY-2.0 | extra parallel pairs for alignment |
| STS Benchmark (`sentence-transformers/stsb`) | CC-BY-SA-4.0 (STS-B as distributed in GLUE/SentEval; sources are news, captions and forums) | paraphrase-stability benchmark (pairs scored >= 4.5/5) |
| Wikipedia 20231101 (`wikimedia/wikipedia`) | CC-BY-SA-3.0 / GFDL | corpus for the canonical space, codebooks and labels; open-data release |
| Samanantar (`ai4bharat/samanantar`) | **CC-BY-NC-4.0** (Hub metadata) | **not used** (non-commercial) |
| PAWS-X | "other" (custom terms) | not used |

### ShareAlike note

FLORES, SIB-200 and Wikipedia are CC-BY-SA. Sphota redistributes *derived statistics*
(centroids, projection matrices) and short *label keyphrases*, plus, in the open-data
release, Wikipedia sentences with their codes. The open-data release is published under
CC-BY-SA-4.0 with attribution, as required. The frozen codebook and projection are
released under Apache-2.0 *and* attributed to these sources; if your legal review treats
the centroids as adaptations of CC-BY-SA material, they are also available under
CC-BY-SA-4.0 (dual statement in `release/v1/NOTICE`).

## JavaScript (Meaning Atlas)

| Package | License |
|---|---|
| `@huggingface/transformers` (transformers.js) | Apache-2.0 |
| `vite` | MIT |
| `typescript` | Apache-2.0 |
