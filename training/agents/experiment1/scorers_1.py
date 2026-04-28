from mlflow.genai.scorers import (
Completeness,
Correctness,
RelevanceToQuery,
)

JUDGE_MODEL="gemini:/gemini-3.1-flash-lite-preview"

MLFLOW_SCORERS = [
    Completeness(model=JUDGE_MODEL),
    Correctness(model=JUDGE_MODEL),
    RelevanceToQuery(model=JUDGE_MODEL),
]