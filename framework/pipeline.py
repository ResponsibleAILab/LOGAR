import json

import pandas as pd

from framework.stage1_extraction import extract_phi_with_reflection
from framework.stage2_replacements import generate_replacements_with_reflection
from framework.stage3_rewrite import rewrite_note_with_reflection
from framework.stage4_final_judge import run_judge


def process_note(
    soap_note: str,
    phi_categories: str,
    rules: str,
    session,
    worker_model: str = "llama3.1-8b",
    judge_model: str = "openai-gpt-5.1",
    max_retries: int = 2,
    max_pipeline_retries: int = 2,
    judge_pass_score: int = 7,
) -> tuple:
    prior_critique = ""
    cleaned_note = ""
    extracted_phi = []
    replacement_map = {}
    messages = []
    history = []

    total_runs = max_pipeline_retries + 1

    for run in range(1, total_runs + 1):
        run_label = "Initial run" if run == 1 else f"Retry {run - 1}/{max_pipeline_retries} (history cleared)"
        print(f"\n  {'─' * 50}")
        print(f"  Pipeline {run_label}")
        if prior_critique:
            print("  Critique from previous run injected into Stage 1.")
        print(f"  {'─' * 50}")

        extracted_phi, messages = extract_phi_with_reflection(
            soap_note=soap_note,
            phi_categories=phi_categories,
            session=session,
            worker_model=worker_model,
            max_retries=max_retries,
            prior_critique=prior_critique,
            judge_pass_score=judge_pass_score,
        )

        replacement_map, messages = generate_replacements_with_reflection(
            messages=messages,
            rules=rules,
            session=session,
            worker_model=worker_model,
            max_retries=max_retries,
            judge_pass_score=judge_pass_score,
        )

        cleaned_note, messages = rewrite_note_with_reflection(
            messages=messages,
            original_note=soap_note,
            replacement_map=replacement_map,
            session=session,
            worker_model=worker_model,
            max_retries=max_retries,
            judge_pass_score=judge_pass_score,
        )

        passed, score, critique = run_judge(
            cleaned_note=cleaned_note,
            session=session,
            judge_model=judge_model,
            judge_pass_score=judge_pass_score,
            rules=rules,
        )

        if passed:
            print(f"  ✓ Accepted after pipeline run {run}.")
            break

        if run < total_runs:
            print(f"  ✗ Judge FAIL (score={score}). Clearing history and restarting pipeline.")
            prior_critique = critique
            history.append(messages)
            messages = []
        else:
            print("  ✗ All pipeline retries exhausted. Keeping best available output.")

    return cleaned_note, extracted_phi, replacement_map, messages, history


def run_pipeline(
    df: pd.DataFrame,
    phi_categories: str,
    rules: str,
    session,
    start: int = 0,
    end: int = 10,
    worker_model: str = "llama3.1-8b",
    judge_model: str = "openai-gpt-5.1",
    max_retries: int = 2,
    max_pipeline_retries: int = 2,
    judge_pass_score: int = 7,
) -> pd.DataFrame:
    if "Cleaned" not in df.columns:
        df["Cleaned"] = None
    df["messages"] = None
    df["phi_extracted"] = None
    df["replacement_map"] = None
    df["history"] = None

    for i in range(start, end):
        print(f"\n{'=' * 60}")
        print(f"Processing note {i + 1} of {end}")
        print(f"{'=' * 60}")

        cleaned_note, extracted_phi, replacement_map, messages, history = process_note(
            soap_note=df["PII_NOTE"][i],
            phi_categories=phi_categories,
            rules=rules,
            session=session,
            worker_model=worker_model,
            judge_model=judge_model,
            max_retries=max_retries,
            max_pipeline_retries=max_pipeline_retries,
            judge_pass_score=judge_pass_score,
        )

        df.at[i, "phi_extracted"] = json.dumps(extracted_phi)
        df.at[i, "replacement_map"] = json.dumps(replacement_map)
        df.at[i, "Cleaned"] = cleaned_note
        df.at[i, "messages"] = messages
        df.at[i, "history"] = history

        print(f"\n✓ Note {i + 1} complete.")
    return df
