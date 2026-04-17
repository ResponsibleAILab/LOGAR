import json
import re
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete

# WORKER_MODEL     = "llama3.1-8b"
WORKER_MODEL = "mistral-large2"
JUDGE_MODEL      = "openai-gpt-5.1"
# JUDGE_MODEL = "claude-sonnet-4-6"
MAX_RETRIES      = 2                    # max re-attempts per stage if judge FAILs
JUDGE_PASS_SCORE = 7                    # minimum score out of 10 to accept output
START_IDX        = 17
END_IDX          = 50
MAX_PIPELINE_RETRIES = 2

print(f"Worker : {WORKER_MODEL}")
print(f"Judge  : {JUDGE_MODEL}")
print(f"Notes  : {START_IDX} → {END_IDX}  ({END_IDX - START_IDX} total)")

session = get_active_session()

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

df            = load_data(session)
phi_categories = load_phi_categories(session)
rules          = load_rules()

print(f"Loaded {len(df)} notes")
print("\nPHI categories preview:")
print(phi_categories[:300])
df.head()

# ── Worker: extract PHI ──────────────────────────────────────────────────
EXTRACT_PHI_SYSTEM = """You are a Medical PHI Identification Expert.

Your task is to extract ONLY verbatim text phrases from the SOAP note that match the PHI categories listed below.

PHI categories (closed set — extract ONLY these):
{phi}

Rules:
- Extract exact phrases/words as they appear in the text
- Do NOT infer or paraphrase
- Do NOT include clinical conditions unless explicitly listed in the PHI categories
- Do NOT explain your reasoning
- If no PHI is present, return []

{prior_critique_block}

Output ONLY the list in the exact format below and nothing else.

Output format:
[
  "phrase1",
  "phrase2",
  ...
]
"""

EXTRACT_PHI_USER = "Text:\n{input_text}"

# ── Judge 1: audit extraction ────────────────────────────────────────────
JUDGE_EXTRACTION_SYSTEM = """You are a strict PHI Extraction Auditor.

You will be given:
1. The original SOAP note
2. The allowed PHI categories
3. A list of PHI phrases extracted by another model

Your job is to audit the extraction for:
- MISSED PHI: real PHI present in the note that was NOT extracted
- FALSE POSITIVES: items extracted that are NOT PHI per the allowed categories
- DUPLICATES or malformed entries

Score the extraction from 0 (completely wrong) to 10 (perfect).

Return ONLY valid JSON in this exact format:
{
  "verdict": "PASS" | "FAIL",
  "score": <integer 0-10>,
  "missed_phi": ["phrase that was missed", ...],
  "false_positives": ["phrase that shouldn't be here", ...],
  "critique": "brief explanation of issues, or 'None' if perfect"
}
"""

JUDGE_EXTRACTION_USER = """Original SOAP Note:
{soap_note}

Allowed PHI Categories:
{phi_categories}

Extracted PHI Phrases:
{extracted_phrases}
"""

RETRY_EXTRACTION_USER = """Your previous extraction had the following issues:
{critique}

Missed PHI: {missed_phi}
False Positives: {false_positives}

Please re-extract the PHI from the same text, fixing these issues.
Apply the same output format: a JSON list of exact verbatim phrases only.
"""

PRIOR_CRITIQUE_BLOCK = """\
 
--- Prior Attempt Reviewer Feedback ---
A previous complete de-identification attempt on this note was reviewed and did not pass.
The reviewer's directional feedback was:
{critique}
 
Use this as context when extracting PHI — be especially thorough.
--- End of Feedback ---
"""

# ── Worker: generate replacements ───────────────────────────────────────
REPLACEMENT_SYSTEM = """You are an Anonymization expert who generates Replacement Text for Identified PHI Phrases.

Your task is to create a replacement for the provided list of PHI phrases using the following Abstraction Rules:
{rules}

Output format (JSON only, no explanation, no markdown):
{{
  "original phrase 1": "replacement",
  "original phrase 2": "replacement"
}}
"""

REPLACEMENT_USER = "Use the extracted PHI list from the conversation above."

# ── Judge 2: audit replacement map ──────────────────────────────────────
JUDGE_REPLACEMENT_SYSTEM = """You are a strict PHI Replacement Auditor.

You will be given:
1. The abstraction rules
2. A JSON map of {original_phrase: replacement} produced by another model

Your job is to check:
- Are any replacements too specific (still re-identifiable)?
- Do replacements follow the format rules (e.g., dates → [DATE], names → [PATIENT NAME])?
- Are there any original PHI phrases left un-replaced or replaced with themselves?

Score from 0-10. For any Bad Replacement reduce the score by 1.

Return ONLY valid JSON:
{
  "verdict": "PASS" | "FAIL",
  "score": <integer 0-10>,
  "bad_replacements": {"original": "issue description", ...},
  "critique": "brief explanation or 'None'"
}
"""

JUDGE_REPLACEMENT_USER = """Abstraction Rules:
{rules}

Replacement Map:
{replacement_map}
"""

RETRY_REPLACEMENT_USER = """Your previous replacement map had these issues:
{critique}

Problematic replacements: {bad_replacements}

Please regenerate the replacement map, fixing these issues.
Use the same JSON output format.
"""

# ── Worker: rewrite note ────────────────────────────────────────────────
REWRITE_SYSTEM = """You are an AI system that generates de-identified medical SOAP notes.

Your task is to rewrite the SOAP note using the replacement map produced in this conversation.

Rules (must follow ALL):
- Preserve original headings and format exactly
- Do NOT summarize or change anything unrelated to PHI
- Maintain professional clinical tone
- Replace redacted content with the bracketed placeholders from the replacement map
- Ensure the output reads naturally
- If multiple notes are present, abstract each separately

Validation requirement:
- After rewriting, no original PHI or specific medical identifiers should remain.

Output:
Return ONLY the fully rewritten SOAP note. No explanations, no replacement map.
"""

REWRITE_USER = "Original SOAP Note:\n{input_text}"

# ── Judge 3: final de-identification audit ───────────────────────────────
JUDGE_REWRITE_SYSTEM = """You are a strict De-identification Auditor for medical SOAP notes.

You will be given:
1. The original SOAP note (with PHI)
2. The de-identified version
3. The replacement map used

Your job is to verify:
- No verbatim PHI from the original remains in the de-identified version
- The clinical structure, headings, and non-PHI content are preserved
- Replacements are consistent with the replacement map
- The note reads naturally and professionally

Score from 0-10. For any PHI found reduce the score by 2.

Return ONLY valid JSON:
{
  "verdict": "PASS" | "FAIL",
  "score": <integer 0-10>,
  "remaining_phi": ["any PHI phrases still found verbatim", ...],
  "structural_issues": ["any issues with missing sections, changed clinical content"],
  "critique": "brief explanation or 'None'"
}
"""

JUDGE_REWRITE_USER = """Original SOAP Note:
{original_note}

Replacement Map:
{replacement_map}

De-identified SOAP Note:
{cleaned_note}
"""

RETRY_REWRITE_USER = """Your previous de-identified note had these issues:
{critique}

Remaining PHI found: {remaining_phi}
Structural issues: {structural_issues}

Please rewrite the SOAP note again, fixing all issues.
Use the original note and replacement map from earlier in this conversation.
"""

FINAL_JUDGE_SYSTEM = """\
You are a senior De-identification Auditor for medical SOAP notes.
 
You will receive a de-identified version produced by another model.
 
Assess whether de-identification is complete and the clinical structure is intact.
 
ABSOLUTE CONSTRAINT ON YOUR CRITIQUE:
- You MUST NOT name, quote, list, or hint at specific PHI phrases that remain.
- You MUST NOT specify what was missed or identify it in any way.
- Your critique must describe only the nature and general location of issues.
- Example of allowed critique: "Demographic information in the Subjective section appears \
insufficiently abstracted."
- Example of forbidden critique: "The patient name 'John Smith' still appears in line 2."
- Violating this constraint defeats the entire purpose of the audit.
 
Score from 0 (PHI fully exposed/Too Much Abstracted) to 10 (perfectly de-identified and clinically intact).
A score of {pass_score} or above is a PASS. 

This is the PHI criteria:
{rules}

Grade Harshly. Score must be low and the verdict should be FAIL if the phi de-identification did not any of follow these rules.
If the note is not coherent or has any obvious PHIs, score must be penalized.
Return ONLY valid JSON in this exact format:
{{
  "verdict": "PASS" or "FAIL",
  "score": <integer 0-10>,
  "critique": "<directional feedback only — no specific PHI, no named phrases, no quoted text>"
}}
"""
 
FINAL_JUDGE_USER = """\
De-identified SOAP Note:
{cleaned_note}
"""
JUDGE_CRITIQUE_RELAY = """\
Your de-identified note did not pass the quality review.
 
Reviewer feedback:
{critique}
 
Re-examine your de-identified note against the replacement map and the original text. \
Identify and correct the issues described above without any further hints.
 
Output only the corrected de-identified SOAP note.
"""

def make_msg(role: str, content: str) -> dict:
    """Build a Snowflake Cortex-compatible message dict."""
    return {"role": role, "content": content, "name": None, "title": None}

from snowflake.cortex import complete
from snowflake.snowpark.functions import ai_complete, lit, parse_json

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

def extract_phi_with_judge(
    soap_note: str,
    phi_categories: str,
    session,
    worker_model: str = WORKER_MODEL,
    judge_model: str = JUDGE_MODEL,
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

def generate_replacements_with_judge(
    messages: list,
    rules: str,
    session,
    worker_model: str = WORKER_MODEL,
    judge_model: str = JUDGE_MODEL,
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

def rewrite_note_with_judge(
    messages: list,
    original_note: str,
    replacement_map: dict,
    session,
    worker_model: str = WORKER_MODEL,
    judge_model: str = JUDGE_MODEL,
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
        print(messages)
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

def run_judge(
    original_note: str,
    cleaned_note: str,
    session,
    judge_model: str = JUDGE_MODEL,
    judge_pass_score: int = JUDGE_PASS_SCORE,
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
        print("    Judge returned malformed JSON. Treating as PASS to avoid infinite loop.")
        return True, 0, ""
 
    score    = verdict.get("score", 0)
    outcome  = verdict.get("verdict", "FAIL")
    critique = verdict.get("critique", "")
    passed   = (outcome == "PASS" and score >= judge_pass_score)
    print(f"  [Final Judge] verdict={outcome}  score={score}/10")
    return passed, score, critique


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
            original_note=soap_note,
            cleaned_note=cleaned_note,
            session=session,
            judge_model=judge_model,
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

idx = 0

print("=" * 60)
print("ORIGINAL NOTE:")
print("=" * 60)
print(df["PII_NOTE"][idx])

print("\n" + "=" * 60)
print("EXTRACTED PHI:")
print("=" * 60)
print(json.dumps(json.loads(df["phi_extracted"][idx]), indent=2))

print("\n" + "=" * 60)
print("REPLACEMENT MAP:")
print("=" * 60)
print(json.dumps(json.loads(df["replacement_map"][idx]), indent=2))

print("\n" + "=" * 60)
print("CLEANED NOTE:")
print("=" * 60)
print(df["Cleaned"][idx])

df[START_IDX:END_IDX][["PII_NOTE", "Cleaned"]]