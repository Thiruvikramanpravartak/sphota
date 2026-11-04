"""Generate tests/vectors/golden.json: anchor vectors + their frozen s1 codes.

Run ONCE when a space version is frozen. Re-running must produce identical codes; CI
(tests/test_core.py::test_golden_codes_are_frozen) enforces this forever.

    python scripts/make_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sphota import from_vector
from sphota.encoders import default_encoder

TEXTS = [
    ("en", "The CEO resigned"),
    ("hi", "सीईओ ने इस्तीफा दे दिया"),
    ("ta", "தலைமை நிர்வாக அதிகாரி ராஜினாமா செய்தார்"),
    ("es", "El director ejecutivo dimitió"),
    ("fr", "Le PDG a démissionné"),
    ("de", "Der Vorstandschef ist zurückgetreten"),
    ("zh", "首席执行官辞职了"),
    ("ja", "CEOが辞任した"),
    ("en", "Heavy rain caused flooding in the city center."),
    ("hi", "भारी बारिश से शहर के केंद्र में बाढ़ आ गई।"),
    ("en", "How do I reset my password?"),
    ("es", "¿Cómo restablezco mi contraseña?"),
    ("en", "The striker scored twice in the final minutes of the match."),
    ("en", "Photosynthesis converts light energy into chemical energy."),
    ("ar", "التمثيل الضوئي يحول الطاقة الضوئية إلى طاقة كيميائية."),
    ("en", "The central bank raised interest rates by half a percentage point."),
    ("ru", "Центральный банк повысил процентные ставки на полпроцента."),
    ("en", "A new species of frog was discovered in the Amazon rainforest."),
    ("pt", "Uma nova espécie de sapo foi descoberta na floresta amazônica."),
    ("en", "Add two cups of flour and stir until smooth."),
    ("bn", "দুই কাপ ময়দা যোগ করুন এবং মসৃণ হওয়া পর্যন্ত নাড়ুন।"),
    ("en", "The museum exhibits paintings from the Renaissance."),
    ("en", "I love you"),
    ("en", "Quantum computers use qubits."),
]


def main() -> None:
    enc = default_encoder()
    X = enc.encode([t for _, t in TEXTS])
    codes = from_vector(X)
    stable = from_vector(X, stable=True)
    items = []
    for (lang, t), x, c, s in zip(TEXTS, X, codes, stable):
        items.append(
            {
                "lang": lang,
                "text": t,
                "code": str(c),
                "code_stable": str(s),
                "label": c.label,
                "confidence": [round(v, 4) for v in c.confidence],
                "vector": [float(v) for v in x.astype(np.float32)],
            }
        )
    out = Path(__file__).resolve().parents[1] / "tests" / "vectors" / "golden.json"
    out.write_text(
        json.dumps(
            {"space": "s1", "encoder": enc.name, "items": items}, ensure_ascii=False, indent=1
        )
    )
    for it in items:
        print(f"{it['code']}  {it['confidence']}  [{it['lang']}] {it['text']}  ::  {it['label']}")


if __name__ == "__main__":
    main()
