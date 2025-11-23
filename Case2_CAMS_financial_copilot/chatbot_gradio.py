
import gradio as gr
import os
from rag_vectorstore import answer_query
import pyttsx3
import uuid
import tempfile

def synthesize_audio(text, filename=None):
    # Use pyttsx3 (offline)
    engine = pyttsx3.init()
    # optional voice settings
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 1.0)
    if filename is None:
        filename = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    # pyttsx3 save to file
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

def chat_fn(query):
    if not query or query.strip()=="":
        return "Please enter a question.", None
    answer, passages = answer_query(query, top_k=4)
    # synthesize audio
    audio_path = synthesize_audio(answer)
    return answer, audio_path

title = "CAMS Financial Analyst Copilot"
description = "Ask questions about the CAMS quarterly result PDF. Answers include short citations and audio."

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}\n\n{description}")
    with gr.Row():
        inp = gr.Textbox(label="Ask a financial question", placeholder="What is the latest quarter EBITDA?")
        submit = gr.Button("Ask")
    out_text = gr.Textbox(label="Answer")
    out_audio = gr.Audio(label="Answer (Audio)")

    def on_ask(q):
        txt, audio = chat_fn(q)
        # return (text, audio filepath)
        return txt, audio

    submit.click(on_ask, inputs=[inp], outputs=[out_text, out_audio])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
