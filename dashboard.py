# dashboard_gradio.py
import os
import gradio as gr
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import json

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URL)
mongo_db = mongo_client["email_parser"]
logs_col = mongo_db["logs"]

def fetch_logs(model_filter="", route_filter="", limit=20):
    """Fetch logs from MongoDB and return as DataFrame for Gradio"""
    q = {}
    if model_filter:
        q["model"] = model_filter
    if route_filter:
        q["route"] = route_filter

    cursor = logs_col.find(q).sort("ts", -1).limit(limit)
    rows = []
    for doc in cursor:
        rows.append({
            "Time": doc.get("ts").strftime("%Y-%m-%d %H:%M:%S"),
            "Route": doc.get("route"),
            "Model": doc.get("model"),
            "Success": doc.get("success"),
            "Usage": json.dumps(doc.get("usage", {}), ensure_ascii=False),
            "Metadata": json.dumps(doc.get("metadata", {}), ensure_ascii=False),
        })
    if not rows:
        return pd.DataFrame([{"Message": "No logs found"}])
    return pd.DataFrame(rows)

with gr.Blocks(title="Logs Dashboard", css="""
    table { font-size: 12px; }
    th, td { padding: 4px 6px !important; white-space: pre-wrap; max-width: 400px; }
""") as demo:
    gr.Markdown("## 📊 Server Usage Dashboard")

    with gr.Row():
        model_in = gr.Textbox(label="Model Filter", placeholder="e.g. gpt-5-mini")
        route_in = gr.Dropdown(
            label="Route Filter",
            choices=["", "/v1/prefill", "/v1/chat/completions"],
            value=""
        )
        limit_in = gr.Slider(5, 100, value=20, step=5, label="Limit (rows)")

    refresh_btn = gr.Button("🔄 Refresh Logs")
    logs_out = gr.DataFrame(label="Logs", wrap=True)

    refresh_btn.click(
        fetch_logs,
        inputs=[model_in, route_in, limit_in],
        outputs=logs_out
    )

    demo.load(fetch_logs, inputs=[model_in, route_in, limit_in], outputs=logs_out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8092)
