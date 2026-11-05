"""60-second quick start: encode text in any language into a Sphota code."""

from sphota import encode, explain, similarity

c = encode("The CEO resigned")
print(repr(c))  # Sphota('JYZA-7M-X0-A8')
print(c.label)  # human-readable label of the region
print([round(p, 2) for p in c.confidence])  # cumulative confidence per level

# The same meaning in Hindi and Spanish lands in the same region (or a nearby one)
for text in ["सीईओ ने इस्तीफा दे दिया", "El director ejecutivo dimitió"]:
    other = encode(text)
    print(
        f"{other}  shared levels with English: {c.common_prefix(other)}  similarity {c.similarity(other):.2f}"
    )

# GPS-style precision: keep only the levels we are confident about
print(encode("The CEO resigned", min_confidence=0.6))

# Alternates for near-boundary inputs, and the k most likely full codes
print(c.alternates[:2])
print(c.candidates(3))

# Codes are plain strings: store them anywhere, compare them without a model
print(similarity(str(c), str(c.prefix(2))), explain(str(c.prefix(1)))["label"])
