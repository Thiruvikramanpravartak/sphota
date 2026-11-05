"""Privacy by depth: share only as many levels as needed. Codes reveal topics; they are not anonymization."""

from sphota import encode

text = "My doctor says my blood sugar is too high."
for level in (1, 2, 4):
    c = encode(text, max_level=level)
    print(f"max_level={level}: {c}  ({c.label})")
print("L1 is a broad topic; L4 is close to a fingerprint of the sentence's meaning.")
