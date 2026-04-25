from mlflow.genai.scorers import (
Correctness,
Completeness,
RelevanceToQuery,
)

JUDGE_MODEL="gemini:/gemini-3.1-flash-lite-preview"

MLFLOW_SCORERS = [
    Correctness(model=JUDGE_MODEL),
    RelevanceToQuery(model=JUDGE_MODEL),
    Completeness(model=JUDGE_MODEL),
]