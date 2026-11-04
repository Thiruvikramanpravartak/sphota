"""Gradio demo for a Python Hugging Face Space (sdk: gradio). Not deployed from CI.

pip install "sphota[encode]" gradio && python hf_space/app.py
"""

import gradio as gr

from sphota import encode


def run(text: str, min_conf: float, lang: str):
    c = encode(text, min_confidence=min_conf or None)
    alts = "\n".join(f"{a}  p={p:.2f}  {a.label_in(lang)}" for a, p in c.alternates)
    conf = " · ".join(f"L{i + 1} {p:.2f}" for i, p in enumerate(c.confidence))
    return str(c), c.label_in(lang), conf, alts or "(none)"


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Text (any language)", value="The CEO resigned"),
        gr.Slider(0, 0.95, value=0, step=0.05, label="min_confidence"),
        gr.Dropdown(
            ["en", "hi", "ta", "bn", "es", "fr", "de", "pt", "zh", "ja", "ar", "ru"],
            value="en",
            label="Label language",
        ),
    ],
    outputs=[
        gr.Textbox(label="Sphota code"),
        gr.Textbox(label="Label"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Alternates"),
    ],
    title="Sphota: GPS for meaning",
    description="Any text, any language → the same short hierarchical code.",
)

if __name__ == "__main__":
    demo.launch()
