import json
from snowflake.snowpark.context import get_active_session

from utils.helper_functions import load_data, load_phi_categories, load_rules
from framework import run_pipeline

session = get_active_session()

WORKER_MODEL     = "llama3.1-8b"
# WORKER_MODEL = "mistral-large2"
JUDGE_MODEL      = "openai-gpt-5.1"
# JUDGE_MODEL = "claude-sonnet-4-6"
MAX_RETRIES      = 2                    # max re-attempts per stage if judge FAILs
JUDGE_PASS_SCORE = 7                    # minimum score out of 10 to accept output
START_IDX        = 0
END_IDX          = 10
MAX_PIPELINE_RETRIES = 2

df            = load_data(session)
phi_categories = load_phi_categories(session)
rules          = load_rules()

if __name__ == "__main__":
    print(f"Worker : {WORKER_MODEL}")
    print(f"Judge  : {JUDGE_MODEL}")
    print(f"Notes  : {START_IDX} → {END_IDX}  ({END_IDX - START_IDX} total)")

    df = run_pipeline(
        df=df,
        phi_categories=phi_categories,
        rules=rules,
        session=session,
        start=START_IDX,
        end=END_IDX,
        worker_model=WORKER_MODEL,
        judge_model=JUDGE_MODEL,
        max_retries=MAX_RETRIES,
        max_pipeline_retries=MAX_PIPELINE_RETRIES,
        judge_pass_score=JUDGE_PASS_SCORE,
    )

    print(df[START_IDX:END_IDX][["PII_NOTE", "Cleaned"]])