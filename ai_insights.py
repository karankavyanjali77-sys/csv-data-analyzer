"""
Groq LLM integration for natural-language dataset analysis.
Completely decoupled from Streamlit — returns plain dicts.
"""

import pandas as pd
import requests
import json
from typing import Optional


def _build_dataset_context(df: pd.DataFrame) -> str:
    """Build a compact textual description of the dataset for the LLM prompt."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    stats_lines = []
    for col in num_cols[:10]:   # cap at 10 to stay within token limits
        s = df[col].describe()
        stats_lines.append(
            f"  - {col}: mean={s['mean']:.2f}, std={s['std']:.2f}, "
            f"min={s['min']:.2f}, max={s['max']:.2f}, "
            f"nulls={int(df[col].isna().sum())}"
        )

    for col in cat_cols[:5]:
        top = df[col].value_counts().head(3).to_dict()
        stats_lines.append(
            f"  - {col} (categorical): top values = {top}, "
            f"nulls={int(df[col].isna().sum())}"
        )

    context = (
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Numeric columns ({len(num_cols)}): {', '.join(num_cols[:15])}\n"
        f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}\n"
        f"Total missing values: {int(df.isna().sum().sum())}\n"
        f"Duplicate rows: {int(df.duplicated().sum())}\n\n"
        f"Column statistics:\n" + "\n".join(stats_lines)
    )
    return context


def generate_ai_summary(
    df: pd.DataFrame,
    api_key: str,
    question: Optional[str] = None,
) -> dict:
    """
    Call Groq's LLM to generate an insight summary or answer a specific question.

    Returns:
        {"success": True,  "summary": "<html-safe string>"}
      or
        {"success": False, "error": "<message>"}
    """
    context = _build_dataset_context(df)

    if question:
        user_message = (
            f"Here is a summary of a CSV dataset:\n\n{context}\n\n"
            f"Please answer this specific question about the data:\n{question}\n\n"
            f"Be concise and direct. Use bullet points if listing multiple items."
        )
    else:
        user_message = (
            f"You are a senior data analyst. Analyse this CSV dataset and provide "
            f"a structured, insightful summary in HTML (use <b>, <ul>, <li>, <p> tags).\n\n"
            f"Dataset context:\n{context}\n\n"
            f"Your summary should include:\n"
            f"1. <b>What this dataset likely represents</b> (inferred from column names)\n"
            f"2. <b>Key patterns and trends</b> observed in the numeric data\n"
            f"3. <b>Data quality issues</b> (missing values, potential outliers)\n"
            f"4. <b>Top 3 actionable recommendations</b> for further analysis\n\n"
            f"Keep it professional and concise — 200-300 words max."
        )

    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": 600,
        "temperature": 0.4,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data    = response.json()
        summary = data["choices"][0]["message"]["content"].strip()
        return {"success": True, "summary": summary}

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        if status == 401:
            return {"success": False, "error": "Invalid API key. Check your Groq key."}
        elif status == 429:
            return {"success": False, "error": "Rate limit hit. Wait a moment and retry."}
        return {"success": False, "error": f"HTTP {status}: {str(e)}"}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Try again."}

    except Exception as e:
        return {"success": False, "error": str(e)}
