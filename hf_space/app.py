"""Gradio demo for a Python Hugging Face Space (sdk: gradio). Not deployed from CI.

pip install "sphota[encode]" gradio && python hf_space/app.py
"""

import gradio as gr

from sphota import encode


def run(text: str, min_conf: float, lang: str):
    c = encode(text, min_confidence=min_conf or None)
    alts = "\n".join(f"{a}  p={p:.2f}" for a, p in c.alternates)
    conf = " · ".join(f"L{i + 1} {p:.2f}" for i, p in enumerate(c.confidence))
    region = f"{c.region} {c.region_label_in(lang)} ({c.region_confidence:.0%})"
    return str(c), region, conf, alts or "(none)"


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Text (any language)", value="The CEO resigned"),
        gr.Slider(0, 0.95, value=0, step=0.05, label="min_confidence"),
        gr.Dropdown(
            [
                "en",
                "hi",
                "ta",
                "bn",
                "te",
                "kn",
                "ml",
                "gu",
                "or",
                "as",
                "ur",
                "es",
                "fr",
                "de",
                "pt",
                "zh",
                "ar",
                "ru",
            ],
            value="en",
            label="Label language",
        ),
    ],
    outputs=[
        gr.Textbox(label="Sphota code"),
        gr.Textbox(label="Region (broad, cross-language)"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Alternates"),
    ],
    title="Sphota: readable multilingual meaning codes",
    description="Any text, any language → the same short hierarchical code.",
)

if __name__ == "__main__":
    demo.launch()
