from mlflow.genai.scorers import (
Correctness,
ExpectationsGuidelines,
Safety,
RelevanceToQuery,
Guidelines,
)

JUDGE_MODEL="gemini:/gemini-3.1-flash-lite-preview"

MLFLOW_SCORERS = [
    Correctness(model=JUDGE_MODEL),
    RelevanceToQuery(model=JUDGE_MODEL),
    ExpectationsGuidelines(model=JUDGE_MODEL),
]