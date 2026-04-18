import json
import re
import pandas as pd
from snowflake.snowpark.context import get_active_session
# from snowflake.cortex import complete

from utils.helper_functions import load_data, load_phi_categories, load_rules, make_msg, call_llm, parse_json_response, parse_list_response
from prompts.extracter_prompts import EXTRACT_PHI_SYSTEM, EXTRACT_PHI_USER, JUDGE_EXTRACTION_SYSTEM, JUDGE_EXTRACTION_USER, RETRY_EXTRACTION_USER, PRIOR_CRITIQUE_BLOCK
from prompts.replacer_prompts import REPLACEMENT_SYSTEM, REPLACEMENT_USER, JUDGE_REPLACEMENT_SYSTEM, JUDGE_REPLACEMENT_USER, RETRY_REPLACEMENT_USER
from prompts.rewriter_prompts import REWRITE_SYSTEM, REWRITE_USER, JUDGE_REWRITE_SYSTEM, JUDGE_REWRITE_USER, RETRY_REWRITE_USER
from prompts.judge_prompts import FINAL_JUDGE_SYSTEM, FINAL_JUDGE_USER

session = get_active_session()

# WORKER_MODEL     = "llama3.1-8b"
WORKER_MODEL = "mistral-large2"
JUDGE_MODEL      = "openai-gpt-5.1"
# JUDGE_MODEL = "claude-sonnet-4-6"
MAX_RETRIES      = 2                    # max re-attempts per stage if judge FAILs
JUDGE_PASS_SCORE = 7                    # minimum score out of 10 to accept output
START_IDX        = 17
END_IDX          = 50
MAX_PIPELINE_RETRIES = 2

df            = load_data(session)
phi_categories = load_phi_categories(session)
rules          = load_rules()

# Stage 1: Extract PHI
def extract_phi_with_judge(
    soap_note: str,
    phi_categories: str,
    session,
    worker_model: str = WORKER_MODEL,
    max_retries: int = MAX_RETRIES,
    prior_critique: str = "",
) -> tuple:

    critique_block = (
        PRIOR_CRITIQUE_BLOCK.format(critique=prior_critique)
        if prior_critique.strip()
        else ""
    )
    messages = [
        make_msg("system", EXTRACT_PHI_SYSTEM.format(
            phi=phi_categories,
            prior_critique_block=critique_block,
        )),
        make_msg("user", EXTRACT_PHI_USER.format(input_text=soap_note)),
    ]
    extracted = []

    for attempt in range(1, max_retries + 1):
        print(f"  [Extraction] Attempt {attempt}/{max_retries}...")

        raw = call_llm(worker_model, messages, session)
        messages.append(make_msg("assistant", raw))

        try:
            extracted = parse_list_response(raw)
        except (json.JSONDecodeError, ValueError):
            print("    Worker returned malformed list. Retrying...")
            messages.append(make_msg("user", "Your response was not valid JSON. Return only a JSON list."))
            continue

        # ── Run Judge 1 ──────────────────────────────────────────────────
        judge_messages = [
            make_msg("system", JUDGE_EXTRACTION_SYSTEM),
            make_msg("user", JUDGE_EXTRACTION_USER.format(
                soap_note=soap_note,
                phi_categories=phi_categories,
                extracted_phrases=json.dumps(extracted, indent=2),
            )),
        ]
        judge_raw = call_llm(worker_model, judge_messages, session)

        try:
            verdict = parse_json_response(judge_raw)
        except (json.JSONDecodeError, ValueError):
            print("    Judge returned malformed response. Accepting worker output.")
            break

        score   = verdict.get("score", 0)
        outcome = verdict.get("verdict", "FAIL")
        print(f"    Judge verdict: {outcome}  (score={score}/10)")

        if outcome == "PASS" and score >= JUDGE_PASS_SCORE:
            print("    ✓ Extraction accepted.")
            break

        if attempt < max_retries:
            retry_prompt = RETRY_EXTRACTION_USER.format(
                critique=verdict.get("critique", "See issues below."),
                missed_phi=json.dumps(verdict.get("missed_phi", [])),
                false_positives=json.dumps(verdict.get("false_positives", [])),
            )
            messages.append(make_msg("user", retry_prompt))
        else:
            print("    ✗ Max retries reached. Using best available extraction.")

    return extracted, messages


# Stage 2: Generate Replacement Map
def generate_replacements_with_judge(
    messages: list,
    rules: str,
    session,
    worker_model: str = WORKER_MODEL,
    max_retries: int = MAX_RETRIES,
) -> tuple:

    messages[0] = make_msg("system", REPLACEMENT_SYSTEM.format(rules=rules))
    messages = messages + [
        # make_msg("system", REPLACEMENT_SYSTEM.format(rules=rules)),
        make_msg("user", REPLACEMENT_USER),
    ]
    replacement_map = {}

    for attempt in range(1, max_retries + 1):
        print(f"  [Replacement] Attempt {attempt}/{max_retries}...")

        raw = call_llm(worker_model, messages, session)
        messages.append(make_msg("assistant", raw))

        try:
            replacement_map = parse_json_response(raw)
        except (json.JSONDecodeError, ValueError):
            print("    Worker returned malformed JSON. Retrying...")
            messages.append(make_msg("user", "Your response was not valid JSON. Return only a JSON object."))
            continue

        # ── Run Judge 2 ──────────────────────────────────────────────────
        judge_messages = [
            make_msg("system", JUDGE_REPLACEMENT_SYSTEM),
            make_msg("user", JUDGE_REPLACEMENT_USER.format(
                rules=rules,
                replacement_map=json.dumps(replacement_map, indent=2),
            )),
        ]
        judge_raw = call_llm(worker_model, judge_messages, session)

        try:
            verdict = parse_json_response(judge_raw)
        except (json.JSONDecodeError, ValueError):
            print("    Judge returned malformed response. Accepting worker output.")
            break

        score   = verdict.get("score", 0)
        outcome = verdict.get("verdict", "FAIL")
        print(f"    Judge verdict: {outcome}  (score={score}/10)")

        if outcome == "PASS" and score >= JUDGE_PASS_SCORE:
            print("    ✓ Replacements accepted.")
            break

        if attempt < max_retries:
            retry_prompt = RETRY_REPLACEMENT_USER.format(
                critique=verdict.get("critique", "See issues below."),
                bad_replacements=json.dumps(verdict.get("bad_replacements", {})),
            )
            messages.append(make_msg("user", retry_prompt))
        else:
            print("    ✗ Max retries reached. Using best available replacement map.")

    return replacement_map, messages


# Stage 3: Rewrite Note with Replacement Map
def rewrite_note_with_judge(
    messages: list,
    original_note: str,
    replacement_map: dict,
    session,
    worker_model: str = WORKER_MODEL,
    max_retries: int = MAX_RETRIES,
) -> tuple:

    messages[0] = make_msg("system", REWRITE_SYSTEM)
    messages = messages + [
        # make_msg("system", REWRITE_SYSTEM),
        make_msg("user", REWRITE_USER.format(input_text=original_note)),
    ]
    cleaned_note = ""

    for attempt in range(1, max_retries + 1):
        print(f"  [Rewrite] Attempt {attempt}/{max_retries}...")
        # print(messages)
        cleaned_note = call_llm(worker_model, messages, session)
        messages.append(make_msg("assistant", cleaned_note))

        # ── Run Judge 3 ──────────────────────────────────────────────────
        judge_messages = [
            make_msg("system", JUDGE_REWRITE_SYSTEM),
            make_msg("user", JUDGE_REWRITE_USER.format(
                original_note=original_note,
                replacement_map=json.dumps(replacement_map, indent=2),
                cleaned_note=cleaned_note,
            )),
        ]
        judge_raw = call_llm(worker_model, judge_messages, session)

        try:
            verdict = parse_json_response(judge_raw)
        except (json.JSONDecodeError, ValueError):
            print("    Judge returned malformed response. Accepting worker output.")
            break

        score   = verdict.get("score", 0)
        outcome = verdict.get("verdict", "FAIL")
        print(f"    Judge verdict: {outcome}  (score={score}/10)")

        if outcome == "PASS" and score >= JUDGE_PASS_SCORE:
            print("    ✓ Cleaned note accepted.")
            break

        if attempt < max_retries:
            retry_prompt = RETRY_REWRITE_USER.format(
                critique=verdict.get("critique", "See issues below."),
                remaining_phi=json.dumps(verdict.get("remaining_phi", [])),
                structural_issues=json.dumps(verdict.get("structural_issues", [])),
            )
            messages.append(make_msg("user", retry_prompt))
        else:
            print("    ✗ Max retries reached. Using best available cleaned note.")

    return cleaned_note, messages

# Final Judge: Evaluate the cleaned note
def run_judge(
    cleaned_note: str,
    session,
    judge_model: str = JUDGE_MODEL,
    judge_pass_score: int = JUDGE_PASS_SCORE,
    rules: str = rules,
) -> tuple:
    judge_messages = [
        make_msg("system", FINAL_JUDGE_SYSTEM.format(pass_score=judge_pass_score, rules=rules)),
        make_msg("user", FINAL_JUDGE_USER.format(
            cleaned_note=cleaned_note,
        )),
    ]
    judge_raw = call_llm(judge_model, judge_messages, session)
 
    try:
        verdict = parse_json_response(judge_raw)
    except (json.JSONDecodeError, ValueError):
        print("Judge returned malformed JSON. Treating as PASS to avoid infinite loop.")
        return True, 0, ""
 
    score    = verdict.get("score", 0)
    outcome  = verdict.get("verdict", "FAIL")
    critique = verdict.get("critique", "")
    passed   = (outcome == "PASS" and score >= judge_pass_score)
    print(f"  [Final Judge] verdict={outcome}  score={score}/10")
    return passed, score, critique

# Main pipeline function to process a single note
def process_note(
    soap_note: str,
    phi_categories: str,
    rules: str,
    session,
    worker_model: str = WORKER_MODEL,
    judge_model: str = JUDGE_MODEL,
    MAX_RETRIES: int = MAX_RETRIES,
    max_pipeline_retries: int = MAX_PIPELINE_RETRIES,
) -> tuple:
    prior_critique = ""
    cleaned_note   = ""
    extracted_phi  = []
    replacement_map = {}
    messages       = []
    history        = []
 
    # run 0 is the first attempt; runs 1..max_pipeline_retries are retries after FAIL
    total_runs = max_pipeline_retries + 1
 
    for run in range(1, total_runs + 1):
        run_label = "Initial run" if run == 1 else f"Retry {run - 1}/{max_pipeline_retries} (history cleared)"
        print(f"\n  {'─'*50}")
        print(f"  Pipeline {run_label}")
        if prior_critique:
            print(f"  Critique from previous run injected into Stage 1.")
        print(f"  {'─'*50}")
 
        # Stage 1
        extracted_phi, messages = extract_phi_with_judge(
            soap_note=soap_note,
            phi_categories=phi_categories,
            session=session,
            worker_model=worker_model,
            max_retries=MAX_RETRIES,
            prior_critique=prior_critique,          # empty on first run
        )
 
        # Stage 2
        replacement_map, messages = generate_replacements_with_judge(
            messages=messages,
            rules=rules,
            session=session,
            worker_model=worker_model,
            max_retries=MAX_RETRIES,
        )
 
        # Stage 3
        cleaned_note, messages = rewrite_note_with_judge(
            messages=messages,
            original_note=soap_note,
            replacement_map=replacement_map,
            session=session,
            worker_model=worker_model,
            max_retries=MAX_RETRIES,
        )
 
        # Judge evaluates final output
        passed, score, critique = run_judge(
            cleaned_note=cleaned_note,
            session=session,
            judge_model=judge_model,
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
            print(f"  ✗ All pipeline retries exhausted. Keeping best available output.")
 
    return cleaned_note, extracted_phi, replacement_map, messages, history
 
# Main function
def run_pipeline(
    df: pd.DataFrame,
    phi_categories: str,
    rules: str,
    session,
    start: int = START_IDX,
    end: int = END_IDX,
    worker_model: str = WORKER_MODEL,
    judge_model: str = JUDGE_MODEL,
) -> pd.DataFrame:
    if "Cleaned" not in df.columns:
        df["Cleaned"]     = None
    df["messages"]        = None
    df["phi_extracted"]   = None
    df["replacement_map"] = None
    df["history"]         = None
 
    for i in range(start, end):
        print(f"\n{'='*60}")
        print(f"Processing note {i + 1} of {end}")
        print(f"{'='*60}")
 
        cleaned_note, extracted_phi, replacement_map, messages, history = process_note(
            soap_note=df["PII_NOTE"][i],
            phi_categories=phi_categories,
            rules=rules,
            session=session,
            worker_model=worker_model,
            judge_model=judge_model,
        )
 
        df.at[i, "phi_extracted"]   = json.dumps(extracted_phi)
        df.at[i, "replacement_map"] = json.dumps(replacement_map)
        df.at[i, "Cleaned"]         = cleaned_note
        df.at[i, "messages"]        = messages
        df.at[i, "history"]         = history
 
        print(f"\n✓ Note {i + 1} complete.")
    return df

if __name__ == "__main__":
    print(f"Worker : {WORKER_MODEL}")
    print(f"Judge  : {JUDGE_MODEL}")
    print(f"Notes  : {START_IDX} → {END_IDX}  ({END_IDX - START_IDX} total)")

    # print(f"Loaded {len(df)} notes")
    # print("\nPHI categories preview:")
    # print(phi_categories[:300])
    # print(df.head())

    df = run_pipeline(
        df=df,
        phi_categories=phi_categories,
        rules=rules,
        session=session,
        start=START_IDX,
        end=END_IDX,
        worker_model=WORKER_MODEL,
        judge_model=JUDGE_MODEL,
    )

    # idx = START_IDX

    # print("=" * 60)
    # print("ORIGINAL NOTE:")
    # print("=" * 60)
    # print(df["PII_NOTE"][idx])

    # print("\n" + "=" * 60)
    # print("EXTRACTED PHI:")
    # print("=" * 60)
    # print(json.dumps(json.loads(df["phi_extracted"][idx]), indent=2))

    # print("\n" + "=" * 60)
    # print("REPLACEMENT MAP:")
    # print("=" * 60)
    # print(json.dumps(json.loads(df["replacement_map"][idx]), indent=2))

    # print("\n" + "=" * 60)
    # print("CLEANED NOTE:")
    # print("=" * 60)
    # print(df["Cleaned"][idx])

    print(df[START_IDX:END_IDX][["PII_NOTE", "Cleaned"]])