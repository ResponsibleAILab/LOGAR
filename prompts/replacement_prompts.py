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
