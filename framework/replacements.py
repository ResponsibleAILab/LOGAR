import json

from utils.helper_functions import call_llm, make_msg, parse_json_response
from prompts.replacer_prompts import (
    JUDGE_REPLACEMENT_SYSTEM,
    JUDGE_REPLACEMENT_USER,
    REPLACEMENT_SYSTEM,
    REPLACEMENT_USER,
    RETRY_REPLACEMENT_USER,
)


def generate_replacements_with_reflection(
    messages: list,
    rules: str,
    session,
    worker_model: str = "llama3.1-8b",
    max_retries: int = 2,
    judge_pass_score: int = 7,
) -> tuple:
    messages[0] = make_msg("system", REPLACEMENT_SYSTEM.format(rules=rules))
    messages = messages + [
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
            messages.append(
                make_msg(
                    "user",
                    "Your response was not valid JSON. Return only a JSON object.",
                )
            )
            continue

        judge_messages = [
            make_msg("system", JUDGE_REPLACEMENT_SYSTEM),
            make_msg(
                "user",
                JUDGE_REPLACEMENT_USER.format(
                    rules=rules,
                    replacement_map=json.dumps(replacement_map, indent=2),
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
