# PHI Removal with Self Reflection and Judge

This repository contains the code from "Leveraging Local LLMs and Online LLMs Collaboration for HIPAA-Compliant De-identification of Mental Health Clinical Data"

This code contains the PHI de-identification pipeline with self-reflection and LLM as a judge using Snowflake Cortex 

## What this Pipeline does

The pipeline processes notes in four stages:

1. Extract PHI candidates from the original note.
2. Generate replacement mappings based on de-identification rules.
3. Rewrite the note with replacements while preserving structure and readability.
4. Run a final online judge over the cleaned output.

If the final judgement fails, the pipeline retries with critique from the final judge back to Stage 1.

## Folder Structure

Entrypoint script:

- `PHI Removal with Self Reflection and Judge.py`
	- loads session/data/config
	- Runs the complete pipeline
	- prints original and cleaned notes

Framework:

- `framework`
    - `extraction.py`: Extract PHI candidates from the original note.
    - `replacement.py`: Generate replacements
    - `rewrite.py`: Rewrite the note
    - `final_judge.py`: Judge model to evaluate the results
    - `pipeline.py`: Processing pipeline with final functions.
    - `__init__.py`: exports all stage and pipeline functions

Prompt templates:

- `prompts`
    - `extraction_prompts.py`
    - `replacement_prompts.py`
    - `rewrite_prompts.py`
    - `final_judge_prompts.py`

Utilities and assets:

- `utils`
    - `helper_functions.py`: Utility functions
    - `utils/PHI.txt`: Contains list of PHI's
    - `utils/rules.txt`: Contains the `PHI ABSTRACTION MANUAL`

## Data and Snowflake dependencies

The pipeline currently expects:

- A Snowflake Python execution context (Snowpark session).
- Source table `O_MINDLLM.COMMON.PMC` with the `PII_NOTE` column.
- Snowflake stage asset path used by helper code for PHI categories (Can be replaced with local PHI.txt made available in utils folder):
	- `O_MINDLLM.COMMON.PII_STAGE/PHI.txt`

## Configuration knobs

Configured in `PHI Removal with Self Reflection and Judge.py`:

- `WORKER_MODEL`: generation/extraction/rewrite model.
- `JUDGE_MODEL`: final judge model.
- `MAX_RETRIES`: max retries per stage.
- `JUDGE_PASS_SCORE`: minimum score threshold for acceptance.
- `MAX_PIPELINE_RETRIES`: end-to-end retry count after final judge returns fail.
- `START_IDX`, `END_IDX`: note slice processed from the input DataFrame.

## Running the pipeline

1. Open a Snowflake-enabled Python environment.
2. Ensure table/stage assets are accessible.
3. Run:

```bash
python "PHI Removal with Self Reflection and Judge.py"
```


