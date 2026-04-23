"""
app.py - HuggingFace Spaces entry point.
Runs the Gradio demo for Enterprise AP-Env.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.demo import build_demo, CSS
import gradio as gr

demo = build_demo()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    css=CSS,
    theme=gr.themes.Default(font=["Inter", "sans-serif"], primary_hue="blue", neutral_hue="gray"),
)
