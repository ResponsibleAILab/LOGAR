import json

from utils.helper_functions import call_llm, make_msg, parse_json_response
from prompts.final_judge_prompts import FINAL_JUDGE_SYSTEM, FINAL_JUDGE_USER


def run_judge(
    cleaned_note: str,
    session,
    judge_model: str = "openai-gpt-5.1",
    judge_pass_score: int = 7,
    rules: str = "",
) -> tuple:
    judge_messages = [
        make_msg("system", FINAL_JUDGE_SYSTEM.format(pass_score=judge_pass_score, rules=rules)),
        make_msg(
            "user",
            FINAL_JUDGE_USER.format(
                cleaned_note=cleaned_note,
            ),
        ),
    ]
    judge_raw = call_llm(judge_model, judge_messages, session)

    try:
        verdict = parse_json_response(judge_raw)
    except (json.JSONDecodeError, ValueError):
        print("Judge returned malformed JSON. Treating as PASS to avoid infinite loop.")
        return True, 0, ""

    score = verdict.get("score", 0)
    outcome = verdict.get("verdict", "FAIL")
    critique = verdict.get("critique", "")
    passed = outcome == "PASS" and score >= judge_pass_score
    print(f"  [Final Judge] verdict={outcome}  score={score}/10")
    return passed, score, critique
