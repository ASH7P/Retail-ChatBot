# === retail_ai_agent.py ===

from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Union, List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import re
import json
import pandas as pd
import numpy as np
from joblib import load
import gradio as gr

# --- PROMPTS ---
from prompts import ROUTER_PROMPT, SQL_PROMPT, VECTOR_PROMPT, FINAL_PROMPT

# --- CONFIGURE LLM ---
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
LLM = GoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

# --- DATABASE & QDRANT SETUP ---
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///transactions.db")
engine = create_engine(DATABASE_URL, echo=False)

qdrant = QdrantClient(path="./local_qdrant")
VECTOR_SIZE = 9

if "transactions" not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name="transactions",
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

# --- SCALER & ENCODER ---
SCALER = load('scaler.joblib')
ONE_HOT_ENCODER = load('encoder.joblib')

class AllowedExtraModel(BaseModel):
    model_config = {"extra": "allow"}

class LLMResponse(BaseModel):
    method: str
    reason: str
    required_data: AllowedExtraModel
    model_config = {"extra": "allow"}

def extract_json_vectors(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in response.")
    return match.group(0)

def extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in response.")
    return match.group(0)

def run_sql_query(sql: str) -> pd.DataFrame:
    cleaned = re.sub(r"```(?:sql)?\s*([\s\S]*?)```", r"\1", sql, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    return pd.read_sql(text(cleaned), engine)

def process_vectors(vectors: Union[list, dict]) -> np.ndarray:
    if isinstance(vectors, dict):
        vectors = [vectors]
    df = pd.DataFrame(vectors)
    scaled = SCALER.transform(df[['tax_amount','transaction_amount','hour']])
    encoded = ONE_HOT_ENCODER.transform(df[['transaction_type','transaction_status']])
    others = df[['day_of_week','is_weekend']].to_numpy()
    return np.hstack([scaled, others, encoded])

def run_vector_search(query_vector: np.ndarray, filter_dict: dict = None, top_k: int = 5):
    must = []
    if filter_dict:
        for k,v in filter_dict.items():
            must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    flt = Filter(must=must) if must else None
    return qdrant.search(
        collection_name="transactions",
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
        query_filter=flt
    )

def route_query(user_query: str) -> LLMResponse:
    raw = LLM.invoke([SystemMessage(ROUTER_PROMPT), HumanMessage(user_query)])
    js  = extract_json(raw)
    return LLMResponse.model_validate_json(js)

def generate_sql(parsed_response: LLMResponse) -> str:
    payload = json.dumps(parsed_response.required_data.sql, indent=2)
    return LLM.invoke([SystemMessage(SQL_PROMPT), HumanMessage(payload)])

def generate_vector_profiles(parsed_response: LLMResponse) -> tuple[str, dict]:
    vec = parsed_response.required_data.vector
    filters = vec.pop("filters", {})
    payload = json.dumps(vec, indent=2)
    out = LLM.invoke([SystemMessage(VECTOR_PROMPT), HumanMessage(payload)])
    return out, filters

def agent_main(user_query: str, history: List[Dict[str,str]]) -> str:
    parsed = route_query(user_query)
    method, reason = parsed.method, parsed.reason
    retrieved = ""
    if method == "SQL":
        sql_code = generate_sql(parsed)
        df = run_sql_query(sql_code)
        retrieved = "\n".join([
            "-- SQL Query --",
            sql_code,
            "-- Results --",
            df.to_string(index=False)
        ])
    elif method == "Vector":
        raw_rows, filters = generate_vector_profiles(parsed)
        cleaned = json.loads(extract_json_vectors(raw_rows))
        vecs = process_vectors(cleaned)
        lines = []
        for v in vecs:
            hits = run_vector_search(v, filters)
            for h in hits:
                lines.append(f"{h.payload} (score={h.score:.3f})")
        retrieved = "\n".join([
            "-- Vector Profiles --",
            json.dumps(cleaned, indent=2),
            "-- Vector Search Hits --",
            *lines
        ])
    else:
        sql_code = generate_sql(parsed)
        df = run_sql_query(sql_code)
        raw_rows, filters = generate_vector_profiles(parsed)
        cleaned = json.loads(extract_json_vectors(raw_rows))
        vecs = process_vectors(cleaned)
        lines = []
        for v in vecs:
            hits = run_vector_search(v, filters)
            for h in hits:
                lines.append(f"{h.payload} (score={h.score:.3f})")
        retrieved = "\n".join([
            "-- SQL Query --",
            sql_code,
            "-- SQL Results --",
            df.to_string(index=False),
            "-- Vector Profiles --",
            json.dumps(cleaned, indent=2),
            "-- Vector Search Hits --",
            *lines
        ])
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    final_context = "\n".join(filter(None, [
        "Chat History:",
        hist,
        "-- Retrieved Data --",
        retrieved,
        "-- User Question --",
        user_query
    ]))
    answer = LLM.invoke([SystemMessage(FINAL_PROMPT), HumanMessage(final_context)])
    return retrieved + "\n\n" + answer

# === GRADIO INTERFACE ===
history: List[Dict[str,str]] = []

def gradio_chat(user_input):
    global history
    response = agent_main(user_input, history)
    history.append({"role":"User", "content": user_input})
    history.append({"role":"Assistant", "content": response})
    return response

if __name__ == "__main__":
    gr.Interface(
        fn=gradio_chat,
        inputs="text",
        outputs="text",
        title="Retail AI Chat",
        description="Ask anything about your retail transaction data."
    ).launch(share=False, server_name="0.0.0.0", server_port=7860)