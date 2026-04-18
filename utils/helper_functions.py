import json
import re
import pandas as pd
from snowflake.cortex import complete

def load_data(session) -> pd.DataFrame:
    df = session.table("O_MINDLLM.COMMON.TOP100").select("PII_NOTE")
    return df.to_pandas()

def load_phi_categories(session) -> str:
    session.file.get("O_MINDLLM.COMMON.PII_STAGE/PHI.txt", "./")
    with open("PHI.txt", "r") as f:
        raw = f.read()
    return "\n".join(
        line.lstrip("* ").strip()
        for line in raw.splitlines()
        if line.strip().startswith("*")
    )

def load_rules() -> str:
    with open("rules.txt", "r") as f:
        return f.read()
    
def make_msg(role: str, content: str) -> dict:
    """Build a Snowflake Cortex-compatible message dict."""
    return {"role": role, "content": content, "name": None, "title": None}

def call_llm(model: str, messages: list, session) -> str:
    """Call Snowflake Cortex complete and return the response string."""
    return complete(model=model, prompt=messages, session=session)

def parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse a JSON object from an LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)

def parse_list_response(text: str) -> list:
    """Strip markdown fences and parse a JSON list from an LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)