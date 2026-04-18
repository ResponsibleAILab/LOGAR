import json

from utils.helper_functions import call_llm, make_msg, parse_json_response, parse_list_response
from prompts.extracter_prompts import (
    EXTRACT_PHI_SYSTEM,
    EXTRACT_PHI_USER,
    JUDGE_EXTRACTION_SYSTEM,
    JUDGE_EXTRACTION_USER,
    PRIOR_CRITIQUE_BLOCK,
    RETRY_EXTRACTION_USER,
)


def extract_phi_with_reflection(
    soap_note: str,
    phi_categories: str,
    session,
    worker_model: str = "llama3.1-8b",
    max_retries: int = 2,
    prior_critique: str = "",
    judge_pass_score: int = 7,
) -> tuple:
    critique_block = (
        PRIOR_CRITIQUE_BLOCK.format(critique=prior_critique)
        if prior_critique.strip()
        else ""
    )
    messages = [
        make_msg(
            "system",
            EXTRACT_PHI_SYSTEM.format(
                phi=phi_categories,
                prior_critique_block=critique_block,
            ),
        ),
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
            messages.append(
                make_msg(
                    "user",
                    "Your response was not valid JSON. Return only a JSON list.",
                )
            )
            continue

        judge_messages = [
            make_msg("system", JUDGE_EXTRACTION_SYSTEM),
            make_msg(
                "user",
                JUDGE_EXTRACTION_USER.format(
                    soap_note=soap_note,
                    phi_categories=phi_categories,
                    extracted_phrases=json.dumps(extracted, indent=2),
                ),
            ),
        ]
        judge_raw = call_llm(worker_model, judge_messages, session)

        try:
            verdict = parse_json_response(judge_raw)
        except (json.JSONDecodeError, ValueError):
            print("    Judge returned malformed response. Accepting worker output.")
            break

        score = verdict.get("score", 0)
        outcome = verdict.get("verdict", "FAIL")
        print(f"    Judge verdict: {outcome}  (score={score}/10)")

        if outcome == "PASS" and score >= judge_pass_score:
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
