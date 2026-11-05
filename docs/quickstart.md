# Quick start (60 seconds)

```bash
pip install "sphota[encode,cli]"
```

```python
>>> from sphota import encode
>>> c = encode("The CEO resigned")
>>> c
Sphota('JYZA-7M-X0-A8')
>>> c.label
'Corporate Executive'
>>> c.confidence          # cumulative per level: how sure each prefix is
[0.8, 0.78, 0.42, 0.14]
```

The first call downloads the encoder (multilingual-e5-small, ONNX, ~470 MB; or
`SPHOTA_ENCODER=int8` for ~120 MB) into your Hugging Face cache. After that everything works
offline, on CPU.

## The same meaning in other languages

```python
>>> encode("El director ejecutivo dimitió")
Sphota('JYZA-7M-0Q-BS')          # same region, same sub-region
>>> c.common_prefix(_)
2
```

## Precision like GPS accuracy

```python
>>> encode("The CEO resigned", min_confidence=0.6)
Sphota('JYZA-7M')               # only the levels we are confident about
>>> c.alternates                 # near a boundary? here are the other likely codes
[(Sphota('Y55K-7M-6Q-RR'), 0.13), ...]
```

## The CLI

```bash
sphota encode "The CEO resigned" "सीईओ ने इस्तीफा दे दिया" --alternates
sphota explain JYZA-7M
sphota analyze tickets.csv -o report.html
sphota certify --model sentence-transformers/all-MiniLM-L6-v2
```

## Light core: no model at all

Parsing, comparing, distances, neighbours and labels need only numpy:

```python
>>> import sphota
>>> sphota.similarity("JYZA-7M-X0-A8", "JYZA-7M-0Q-BS")
0.98
>>> sphota.neighbors("JYZA", 3)
```

## Offline / air-gapped

```bash
sphota bundle -o sphota-bundle            # on a connected machine
SPHOTA_OFFLINE=1 SPHOTA_BUNDLE=./sphota-bundle sphota encode "hello"   # anywhere
```
