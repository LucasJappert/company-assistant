# backend/frontend.py
import gradio as gr
import requests


def chat(query: str) -> str:
    resp = requests.post("http://127.0.0.1:8000/ask", json={"query": query, "k": 5})
    data = resp.json()
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    return f"{answer}\n\nFuentes: {sources}"


demo = gr.Interface(
    fn=chat, inputs=gr.Textbox(label="Pregunta"), outputs=gr.Textbox(label="Respuesta")
)

if __name__ == "__main__":
    demo.launch()
