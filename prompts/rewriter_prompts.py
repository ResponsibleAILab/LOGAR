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