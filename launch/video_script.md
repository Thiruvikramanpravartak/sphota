# 60-second demo video script

**Format:** screen recording of the Meaning Atlas + terminal, voice-over, captions on.
**Tone:** calm, curious, concrete. No hype words.

| Time | Visual | Voice-over / caption |
|---|---|---|
| 0:00–0:05 | Dark map of glowing "continents" slowly drifting | "This is a map of meaning." |
| 0:05–0:12 | Type *The CEO resigned* → camera flies to a region, code `JYZA-7M-X0-A8` appears with an accuracy ring | "Every sentence has an address. Like GPS, but for what it means." |
| 0:12–0:22 | Compare mode: Hindi *सीईओ ने इस्तीफा दे दिया* or Spanish lands next to it; chip "share 2 of 4 levels" | "Any language. The same meaning lands in the same place." |
| 0:22–0:30 | Zoom out/in: continent labels → region labels → sub-regions | "The code is hierarchical. Shorter prefix, broader topic." |
| 0:30–0:38 | Terminal: `pip install sphota` → `encode(...)` → `Sphota('JYZA-7M-X0-A8')`, `.confidence` | "One line of Python. It runs on a laptop CPU, and in your browser." |
| 0:38–0:48 | Split screen: index built with model A, query with model B; same results | "Switch embedding models without re-indexing. Codes stay frozen; small adapters map new models in." |
| 0:48–0:55 | `sphota analyze tickets.csv` → HTML report scroll | "Group by meaning. Route, cache, dedupe, in any language." |
| 0:55–1:00 | Logo स्फोट + "Open. Frozen. Readable." + URL | "Sphota. GPS for meaning. Open source." |

**Shots to capture:** `npm run dev` in `atlas/` (1920×1080, browser zoom 100%, reduced UI
chrome), terminal with large font (20 pt), `examples/03_topic_analytics.py` output.
