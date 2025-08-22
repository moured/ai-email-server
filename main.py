# uvicorn main:app --host 127.0.0.1 --port 8090 --reload
import os
import csv
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI 
from pymongo import MongoClient

# ---- API Keys -----------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "deepseek/deepseek-r1-0528:free" # "gpt-5-mini"  # "deepseek/deepseek-r1-0528:free"
if not (OPENAI_API_KEY and OPENROUTER_API_KEY):
    raise RuntimeError("You must set either OPENAI_API_KEY or OPENROUTER_API_KEY")

# ---- Initialize Clients -------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
openrouter_client = (
    OpenAI(
        base_url="https://openrouter.ai/api/v1", # openrouter base url
        api_key=OPENROUTER_API_KEY,
    )
)

# ---- Metadata info ------------------------------------------------
CSV_PATH = "data.csv"
CSV_FIELDS = ["amount", "currency", "due_date", "description", "company", "contact"]
# model pricing .. 
MODEL_PRICES = {
    # OpenAI models
    "gpt-5": (1.25, 10.00), 
    "gpt-5-mini": (0.25,  2.00),
    "deepseek/deepseek-r1-0528:free": (0.00, 0.00),
    "moonshotai/kimi-k2:free": (0.00, 0.00),
    "qwen/qwen3-235b-a22b:free": (0.00, 0.00),
}

# ---- Initialize app & db ------------------------------------------
app = FastAPI(title="Email Parsing Server", version="1.0.0")

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URL)
mongo_db = mongo_client["email_parser"]
logs_col = mongo_db["logs"]

# ---- Chat & prefill pydantic templates, good for validation -------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"] = "user"
    content: Any

class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Any] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = 1

    @classmethod # to support gpt-5,4 and old versions 
    def validate(cls, value):
        if isinstance(value, dict) and "max_tokens" in value:
            value["max_completion_tokens"] = value.pop("max_tokens")

        # # validate temperature for gpt-5-mini    
        # if isinstance(value, dict):
        #     model = value.get("model", "")
        #     if model.startswith("gpt-") and "temperature" in value:
        #         value.pop("temperature", None)

        return super().model_validate(value)
    
class PrefillRequest(BaseModel):
    email_text: str = Field(..., min_length=3)
    model: Optional[str] = None

class PrefillResponse(BaseModel):
    success: bool
    message: str

# ---- utils -----------------------------------------------------
def ensure_csv(path: str = CSV_PATH, headers: List[str] = CSV_FIELDS) -> None:
    """Create CSV file with header row if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

def write_csv_row(row: Dict[str, Any], path: str = CSV_PATH) -> None:
    """Append single email metadata row to the CSV"""
    ensure_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        # missing keys -> ""
        safe_row = {k: (row.get(k, "") if row.get(k) is not None else "") for k in CSV_FIELDS}
        writer.writerow(safe_row)


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Try to extract JSON object from the model response.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{\s*(?:.|\n|\r)*\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON object found in model output")


def pick_client_and_model(model: str) -> OpenAI:
    """
    If model starts with gpt-, use OpenAI client; else use OpenRouter client.
    """
    if model.startswith("gpt-"):
        return openai_client
    else:
        return openrouter_client

# mongodb utils
def log_request(route: str, model: str, query: Dict[str, Any], response: Dict[str, Any],
                metadata: Dict[str, Any], success: bool, fingerprint: Optional[str] = None,
                usage: Optional[Dict[str, Any]] = None):
    doc = {
        "route": route,
        "model": model,
        "query": query,
        "response": response,
        "metadata": metadata,
        "success": success,
        "fingerprint": fingerprint,
        "usage": usage,
        "ts": datetime.utcnow()
    }
    try:
        logs_col.insert_one(doc)
    except Exception as e:
        print("DB log failed:", e, flush=True)

def compute_md5(text: str) -> str:
    """Return hex MD5 hash of a text (UTF-8)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def summarize_usage(model: str, usage: dict) -> dict:
    """
    Given a model name and OpenAI 'usage' dict, return tokens + cost.
    usage example: {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
    """
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}

    p = usage.get("prompt_tokens", 0)
    c = usage.get("completion_tokens", 0)
    t = usage.get("total_tokens", p + c)

    in_price, out_price = MODEL_PRICES.get(model, (0.0, 0.0))
    cost = (p * in_price + c * out_price) / 1000.0

    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": t,
        "cost_usd": cost,
    }

# ---- Chat route -----------------------------------------------------
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    """
    Chat endpoint
    """
    payload = req.model_dump(by_alias=True, exclude_none=True)
    model = payload.pop("model", DEFAULT_MODEL)
    payload["model"] = model # looks redundent but sometimes the request does not contain model name.
    client = pick_client_and_model(model)
    payload["stream"] = False

    try:
        resp = client.chat.completions.create(**payload)
        data = resp.model_dump()
        
        usage = data.get("usage")
        summary = summarize_usage(model, usage)
        log_request("/v1/chat/completions", model, payload, data, {}, True, usage=summary)
        return JSONResponse(content=data)
    except Exception as e:
        log_request("/v1/chat/completions", model, payload, {"error": str(e)}, {}, False)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- Routes: Prefill ----------------------------
@app.post("/v1/prefill", response_model=PrefillResponse)
def prefill(req: PrefillRequest):
    """
    Delegate extraction fully to the LLM via strict instructions,
    then append the exact JSON fields to data.csv.
    """
    model = req.model or DEFAULT_MODEL
    client = pick_client_and_model(model)

    fingerprint = compute_md5(req.email_text)

   # Check if fingerprint dublicate (this scenario if same email received to multiple people)
    existing = logs_col.find_one({
        "route": "/v1/prefill",
        "fingerprint": fingerprint,
        "success": True
    })
    if existing:
        cached = existing.get("metadata", {})
        write_csv_row(cached)
        log_request("/v1/prefill", model, {"email_text": req.email_text}, {"cache_hit": True}, cached, True, fingerprint, usage=None)
        return PrefillResponse(
            success=True,
            message="data extracted and returned from cache"
        )
    
    system_prompt = (
        "You are a strict information extractor for payment data in emails.\n"
        "Return ONLY a single JSON object with these exact keys:\n"
        '  - "amount": string (exact numeric as shown, e.g. "1234.56" or "1.234,56")\n'
        '  - "currency": string (ISO like "EUR", "USD", "GBP"; if only symbol is present, map it accordingly)\n'
        '  - "due_date": string (YYYY-MM-DD; null if absent)\n'
        '  - "description": string (short human description; null if unknown)\n'
        '  - "company": string (vendor/company requesting payment; null if unknown)\n'
        '  - "contact": string (person or email to reach; null if unknown)\n'
        "Rules:\n"
        "- Base answers ONLY on the email text; if not clearly present, output null.\n"
        "- No extra fields, no comments, no markdown, no code fences.\n"
        "- Output must be valid JSON."
    )
    user_prompt = "Email text follows. Extract the fields and return ONLY the JSON object.\n\n" + req.email_text

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 1,
    }

    # If using OpenAI models, ask for JSON mode explicitly
    if model.startswith("gpt-"):
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**payload)
        content = resp.choices[0].message.content
        data = _extract_json(content)
        resp_dump = resp.model_dump()
        usage = resp_dump.get("usage")

        # ensure all required keys exist
        for k in CSV_FIELDS:
            data.setdefault(k, None)

        write_csv_row(data)
        summary = summarize_usage(model, usage)
        log_request("/v1/prefill", model, {"email_text": req.email_text}, resp_dump, data, True, 
                    fingerprint, usage=summary)
        return PrefillResponse(success=True, message="data extracted and written")
    except Exception as e:
        log_request("/v1/prefill", model, {"email_text": req.email_text}, {"error": str(e)}, {},
                    False, fingerprint)
        return PrefillResponse(success=False, message=str(e))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    # Run: uvicorn main:app --host 127.0.0.1 --port 8090
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8090, reload=True)
