"""Core logic for PS69 - Customer Feedback Classifier (Groq / Gemini)."""
import json
import os
import re

import pandas as pd

CATEGORIES = ["Delivery", "Product Quality", "Price", "Customer Service", "Website/App", "Other"]
PRIORITIES = ["High", "Medium", "Low"]
PROVIDER_ENV = os.getenv("LLM_PROVIDER", "").lower()   # "groq" or "gemini" (auto-detected from key)
GROQ_PREFERRED = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
GEMINI_PREFERRED = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-2.5-flash"]
_model_cache = {}


PROMPT = """You are a customer feedback analyst.
Classify each feedback item below.

Rules:
- category must be exactly one of: {categories}
- priority must be exactly one of: High, Medium, Low
  (High = safety issue, refund/legal threat, repeated failure or very angry customer;
   Medium = clear complaint; Low = praise, suggestion or minor remark)
- sentiment must be Positive, Neutral or Negative
- reason: max 10 words explaining the choice

Return ONLY a JSON object in this format, with one entry per item:
{{"results": [{{"id": 0, "category": "...", "priority": "...", "sentiment": "...", "reason": "..."}}]}}

Feedback items:
{items}
"""


def build_prompt(batch):
    """batch: list of (id, text) tuples."""
    items = "\n".join(f"{i}. {t}" for i, t in batch)
    return PROMPT.format(categories=", ".join(CATEGORIES), items=items)


def parse_response(text):
    """Parse the LLM reply into a list of dicts. Tolerates ```json fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = data["results"] if "results" in data else [data]
    return data


def _normalise(row):
    cat = row.get("category", "Other")
    pri = row.get("priority", "Low")
    cat = next((c for c in CATEGORIES if c.lower() == str(cat).lower()), "Other")
    pri = next((p for p in PRIORITIES if p.lower() == str(pri).lower()), "Low")
    return {
        "id": int(row["id"]),
        "category": cat,
        "priority": pri,
        "sentiment": str(row.get("sentiment", "Neutral")).title(),
        "reason": str(row.get("reason", "")),
    }


def detect_provider(api_key):
    if PROVIDER_ENV in ("groq", "gemini"):
        return PROVIDER_ENV
    return "groq" if api_key.startswith("gsk_") else "gemini"


def _retry(fn):
    """Retry on rate limit / server busy errors with increasing waits."""
    import time
    last = None
    for attempt in range(5):
        try:
            return fn()
        except Exception as e:                      # noqa: BLE001
            msg = str(e)
            if any(c in msg for c in ("429", "500", "502", "503", "504", "overloaded", "UNAVAILABLE")):
                last = e
                time.sleep(2 * (attempt + 1))
            else:
                raise
    raise last


def _pick(available, preferred, fallback_filter):
    for p in preferred:
        if p in available:
            return p
    rest = [m for m in available if fallback_filter(m)]
    if not rest:
        raise RuntimeError(f"No usable model found. Available: {available[:10]}")
    return rest[0]


def call_groq(prompt, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    if "groq" not in _model_cache:      # ask the API which models exist, so retired names never break us
        ids = [m.id for m in client.models.list().data]
        _model_cache["groq"] = _pick(ids, GROQ_PREFERRED,
                                     lambda m: "llama" in m and not any(x in m for x in ("guard", "whisper", "tts")))
    model = _model_cache["groq"]
    resp = _retry(lambda: client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]))
    return resp.choices[0].message.content


def call_gemini(prompt, api_key):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    if "gemini" not in _model_cache:
        ids = [m.name.replace("models/", "") for m in client.models.list()
               if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]
        _model_cache["gemini"] = _pick(ids, GEMINI_PREFERRED, lambda m: "flash" in m and "image" not in m)
    model = _model_cache["gemini"]
    resp = _retry(lambda: client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json")))
    return resp.text


def call_llm(prompt, api_key):
    return call_groq(prompt, api_key) if detect_provider(api_key) == "groq" else call_gemini(prompt, api_key)


def used_model():
    return _model_cache.get("groq") or _model_cache.get("gemini") or "unknown"


def classify_dataframe(df, text_col, api_key, batch_size=10, llm=call_llm, progress=None):
    """Classify every row of df[text_col]. Returns df with new columns.
    `llm` can be swapped for a stub in tests."""
    texts = df[text_col].fillna("").astype(str).tolist()
    results = {}
    for start in range(0, len(texts), batch_size):
        batch = list(enumerate(texts[start:start + batch_size], start=start))
        prompt = build_prompt(batch)
        for attempt in range(2):  # one retry if JSON is malformed
            try:
                rows = parse_response(llm(prompt, api_key))
                for r in rows:
                    n = _normalise(r)
                    results[n["id"]] = n
                break
            except (json.JSONDecodeError, KeyError, ValueError):
                if attempt == 1:
                    for i, _ in batch:
                        results.setdefault(i, {"id": i, "category": "Other", "priority": "Low",
                                               "sentiment": "Neutral", "reason": "parse error"})
        if progress:
            progress(min(start + batch_size, len(texts)) / len(texts))
    out = df.copy()
    for col in ["category", "priority", "sentiment", "reason"]:
        out[col] = [results.get(i, {}).get(col, "") for i in range(len(texts))]
    return out
