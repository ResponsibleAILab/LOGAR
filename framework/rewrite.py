import json

from utils.helper_functions import call_llm, make_msg, parse_json_response
from prompts.rewrite_prompts import (
    JUDGE_REWRITE_SYSTEM,
    JUDGE_REWRITE_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
    RETRY_REWRITE_USER,
)


def rewrite_note_with_reflection(
    messages: list,
    original_note: str,
    replacement_map: dict,
    session,
    worker_model: str = "llama3.1-8b",
    max_retries: int = 2,
    judge_pass_score: int = 7,
) -> tuple:
    messages[0] = make_msg("system", REWRITE_SYSTEM)
    messages = messages + [
        make_msg("user", REWRITE_USER.format(input_text=original_note)),
    ]
    cleaned_note = ""

    for attempt in range(1, max_retries + 1):
        print(f"  [Rewrite] Attempt {attempt}/{max_retries}...")
        cleaned_note = call_llm(worker_model, messages, session)
        messages.append(make_msg("assistant", cleaned_note))

        judge_messages = [
            make_msg("system", JUDGE_REWRITE_SYSTEM),
            make_msg(
                "user",
                JUDGE_REWRITE_USER.format(
                    original_note=original_note,
                    replacement_map=json.dumps(replacement_map, indent=2),
                    cleaned_note=cleaned_note,
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
