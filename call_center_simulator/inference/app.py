"""Gradio UI for call-center simulator."""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def generate_reply(
    situation: str,
    history_text: str,
    openness: float,
    conscientiousness: float,
    extraversion: float,
    agreeableness: float,
    neuroticism: float,
    max_tokens: int,
) -> str:
    """Call FastAPI /generate and return the reply."""
    history = []
    for line in history_text.strip().splitlines():
        if ": " in line:
            role, text = line.split(": ", 1)
            history.append({"role": role.strip(), "text": text.strip()})

    payload = {
        "history": history,
        "situation": situation,
        "ocean_profile": {
            "openness": openness,
            "conscientiousness": conscientiousness,
            "extraversion": extraversion,
            "agreeableness": agreeableness,
            "neuroticism": neuroticism,
        },
        "max_new_tokens": max_tokens,
    }
    try:
        response = httpx.post(f"{API_URL}/generate", json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()["reply"]
    except Exception as exc:
        return f"Error: {exc}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Call-Center Simulator") as demo:
        gr.Markdown(
            "# Call-Center Simulator\nGenerate client replies with OCEAN personality."
        )
        with gr.Row():
            with gr.Column():
                situation = gr.Textbox(
                    label="Situation",
                    lines=2,
                    placeholder="Client calls about delayed delivery...",
                )
                history_text = gr.Textbox(
                    label="Dialog history (role: text per line)",
                    lines=5,
                    placeholder="operator: Hello, how can I help?",
                )
                gr.Markdown("### OCEAN Profile")
                openness = gr.Slider(0.0, 1.0, value=0.5, label="Openness (O)")
                conscientiousness = gr.Slider(
                    0.0, 1.0, value=0.5, label="Conscientiousness (C)"
                )
                extraversion = gr.Slider(0.0, 1.0, value=0.5, label="Extraversion (E)")
                agreeableness = gr.Slider(
                    0.0, 1.0, value=0.5, label="Agreeableness (A)"
                )
                neuroticism = gr.Slider(0.0, 1.0, value=0.5, label="Neuroticism (N)")
                max_tokens = gr.Slider(16, 512, value=128, step=16, label="Max tokens")
                btn = gr.Button("Generate reply", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Client reply", lines=6)
        btn.click(
            generate_reply,
            inputs=[
                situation,
                history_text,
                openness,
                conscientiousness,
                extraversion,
                agreeableness,
                neuroticism,
                max_tokens,
            ],
            outputs=output,
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
