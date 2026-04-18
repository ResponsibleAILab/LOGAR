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

