# agent.py
import ast
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import requests
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

import mlflow
from mlflow.entities import SpanType
import math
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url=os.environ["GEMINI_OPENAI_BASE_URL"],
)
EMBEDDING_MODEL = "gemini-embedding-001"

@dataclass
class AgentConfig:
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    prompt_uri: str = "prompts:/mlflow-agent-system@prod"
    autolog: bool = True


class MLflowDocsAgent:
    """
    Wrapper around a LangChain agent that loads its system prompt from the
    MLflow Prompt Registry and exposes a stable predict() method for
    mlflow.genai.evaluate().

    Tools and retrievers are added imperatively via add_tool / add_retriever,
    so the same class scaffolds Stage 2 → Stage 3 → Stage 4 of the workshop.
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self._tools: list[Callable | BaseTool] = []
        self._retrievers: list[Callable[[str], list[dict]]] = []
        self._agent = None  # built lazily on first predict()

        if self.config.autolog:
            mlflow.langchain.autolog()

    # ── Registration API ─────────────────────────────────────────────────
    def add_tool(self, tool: Callable | BaseTool) -> "MLflowDocsAgent":
        """Register a tool. Returns self so calls can be chained."""
        self._tools.append(tool)
        self._agent = None  # invalidate — needs rebuild
        return self

    def add_tools(self, tools: Sequence[Callable | BaseTool]) -> "MLflowDocsAgent":
        for t in tools:
            self.add_tool(t)
        return self

    def add_retriever(
        self, retriever: Callable[[str], list[dict]]
    ) -> "MLflowDocsAgent":
        """
        Register a retriever. Each retriever takes a query string and
        returns a list of {page_content, metadata} dicts (MLflow's
        SpanType.RETRIEVER convention).
        """
        self._retrievers.append(retriever)
        self._agent = None
        return self

    # ── Build / invoke ───────────────────────────────────────────────────
    def _build(self):
        llm = ChatGoogleGenerativeAI(
            model=self.config.model,
            temperature=self.config.temperature,
        )
        prompt = mlflow.genai.load_prompt(self.config.prompt_uri)
        self._agent = create_agent(
            model=llm,
            tools=self._tools,
            system_prompt=prompt.template,
        )

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve_context")
    def _retrieve(self, query: str) -> str:
        """Run all registered retrievers and concatenate results."""
        if not self._retrievers:
            return ""
        chunks = []
        for r in self._retrievers:
            chunks.extend(r(query))
        # Format for the LLM — MLflow autocaptures this as the retriever output
        return "\n\n".join(c["page_content"] for c in chunks)

    @mlflow.trace(span_type=SpanType.AGENT, name="MLflowDocsAgent")
    def predict(self, question: str) -> str:
        """
        Stable invocation surface. Compatible with mlflow.genai.evaluate's
        predict_fn(inputs: dict) signature when called as agent.predict(**inputs).
        """
        if self._agent is None:
            self._build()

        # Optional retrieval step — empty string if no retrievers registered
        context = self._retrieve(question)
        user_message = (
            f"Context:\n{context}\n\nQuestion: {question}" if context else question
        )

        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        return result["messages"][-1].content




@dataclass
class SemanticRetriever:
    """
    Closure-style retriever: holds the embedded index and config, exposes
    a single __call__(query) -> list[dict] that matches the agent's
    add_retriever() contract.
    """
    index: list[dict]                      # [{"id", "text", "embedding"}, ...]
    embedding_client: OpenAI
    embedding_model: str = "gemini-embedding-001"
    top_k: int = 3
    name: str = "mlflow_docs"              # shows up in trace span names

    def __post_init__(self):
        # Wrap the public entry point in a trace at construction time
        # so the span name reflects this retriever's `name` attribute.
        self.__call__ = mlflow.trace(
            span_type=SpanType.RETRIEVER,
            name=f"retrieve.{self.name}",
        )(self.__call__)

    @mlflow.trace(span_type=SpanType.EMBEDDING)
    def _embed_query(self, query: str) -> list[float]:
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )
        return response.data[0].embedding

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def __call__(self, query: str) -> list[dict]:
        """Retrieve top_k docs for a query in MLflow/LangChain shape."""
        query_embedding = self._embed_query(query)

        scored = [
            {
                "doc": doc,
                "score": self._cosine(query_embedding, doc["embedding"]),
            }
            for doc in self.index
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Convert to the {page_content, metadata} shape the agent expects.
        # Score and doc id go into metadata so they survive into the trace
        # and the eval dataset's retrieved_context column.
        return [
            {
                "page_content": s["doc"]["text"],
                "metadata": {
                    "doc_id": s["doc"]["id"],
                    "score": s["score"],
                },
            }
            for s in scored[: self.top_k]
        ]
    
DOCUMENT_STORE = {
    "doc1": (
        "MLflow Tracing provides end-to-end observability for GenAI applications. "
        "It captures every LLM call, retrieval step, tool invocation, and agent reasoning "
        "as structured spans with full input/output visibility. Traces are logged automatically "
        "via autologging or manually with the @mlflow.trace decorator and mlflow.start_span() context manager."
    ),
    "doc2": (
        "MLflow AI Gateway lets teams manage multiple LLM providers through a single, secure endpoint. "
        "It centralizes access control, cost tracking, and rate limiting across providers like OpenAI, "
        "Anthropic, and Google. Features include traffic routing, automatic fallbacks between providers, "
        "budget alerts, and comprehensive usage analytics."
    ),
    "doc3": (
        "MLflow Evaluation enables systematic testing of GenAI applications using scorers. "
        "Built-in scorers include Correctness, Safety, RelevanceToQuery, and RetrievalGroundedness. "
        "Teams can also create custom scorers with Guidelines() for natural-language criteria "
        "or make_judge() for LLM-as-judge evaluators. Run evaluations with mlflow.genai.evaluate()."
    ),
    "doc4": (
        "MLflow Prompt Registry allows teams to version, share, and manage prompts centrally. "
        "Prompts support template variables, lifecycle aliases (development, staging, production), "
        "and integration with experiments. Use mlflow.genai.register_prompt() to store prompts "
        "and mlflow.genai.load_prompt() to retrieve them by name and version."
    ),
    "doc5": (
        "Judge alignment in MLflow teaches LLM judges to match human evaluation standards. "
        "The workflow is: create a judge with make_judge(), collect human feedback on traces "
        "via mlflow.log_feedback(), then call judge.align() with the SIMBA or GEPA optimizer. "
        "Aligned judges typically show 30-50% reduction in false positives compared to generic prompts."
    ),
    "doc6": (
        "MLflow autologging automatically captures metrics, parameters, and traces for 40+ frameworks "
        "including OpenAI, LangChain, LlamaIndex, DSPy, and AutoGen. Enable it with one line — "
        "e.g., mlflow.openai.autolog() or mlflow.langchain.autolog(). Each call is traced with "
        "token counts, latency, and the full request/response payload."
    ),
    "doc7": (
        "MLflow Deployments serves models as production-ready REST API endpoints. "
        "It supports local inference servers, Kubernetes, AWS SageMaker, and managed platforms. "
        "Models are deployed directly from the Model Registry with version pinning, "
        "and each serving endpoint gets automatic request/response logging."
    ),
    "doc8": (
        "MLflow tracks token usage and cost across LLM applications. Every traced call records "
        "input tokens, output tokens, and total tokens. When provider pricing is configured, "
        "MLflow calculates per-call and cumulative costs. Use mlflow.search_traces() to query "
        "usage patterns and identify expensive operations."
    ),
    "doc9": (
        "MLflow Experiment Tracking organizes ML and GenAI work into experiments and runs. "
        "Each run logs parameters, metrics, artifacts, and traces. Teams use mlflow.set_experiment() "
        "to group related work, mlflow.log_param() and mlflow.log_metric() for tracking, "
        "and the MLflow UI to compare results across runs visually."
    ),
    "doc10": (
        "MLflow is fully open source and vendor-neutral. It works with any cloud provider, "
        "ML framework, or LLM provider without lock-in. The unified API covers the full lifecycle "
        "from experimentation through evaluation to production deployment, with a single tracking "
        "server that stores all artifacts, metrics, and traces."
    ),
}


#Tools
def get_mlflow_version_pypi() -> str:
    """
    Fetches the current stable release of MLflow directly from PyPI.
    No API key required.
    """
    try:
        response = requests.get("https://pypi.org/pypi/mlflow/json", timeout=5)
        response.raise_for_status()
        return response.json()["info"]["version"]
    except Exception as e:
        return f"Could not fetch version from PyPI: {str(e)}"

def get_package_info(package: str, max_versions: int = 5) -> dict:
    """Fetch version history + release notes for a PyPI package."""
    # 1. PyPI metadata
    pypi = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=10).json()

    info = pypi["info"]
    releases = pypi["releases"]

    # 2. Build a timestamped version list (skip yanked/empty releases)
    versions = []
    for version, files in releases.items():
        if not files:
            continue
        upload_time = files[0]["upload_time"]
        versions.append((version, upload_time))

    versions.sort(key=lambda v: v[1], reverse=True)
    recent = versions[:max_versions]

    # 3. Try to find a GitHub repo from project_urls
    urls = info.get("project_urls") or {}
    github_repo = _extract_github_repo(urls)

    # 4. Fetch release notes from GitHub if we found a repo
    notes = {}
    if github_repo:
        gh_url = f"https://api.github.com/repos/{github_repo}/releases"
        gh_releases = requests.get(gh_url, timeout=10).json()
        for r in gh_releases:
            # GitHub tags often prefix with 'v' — normalize both sides
            tag = r["tag_name"].lstrip("v")
            notes[tag] = {
                "name": r["name"],
                "body": r["body"],
                "published_at": r["published_at"],
                "url": r["html_url"],
            }

    # 5. Stitch it together
    return {
        "package": package,
        "latest": info["version"],
        "summary": info["summary"],
        "homepage": info.get("home_page"),
        "project_urls": urls,
        "recent_versions": [
            {
                "version": v,
                "released": ts,
                "notes": notes.get(v),
            }
            for v, ts in recent
        ],
    }


def _extract_github_repo(project_urls: dict) -> str | None:
    """Find 'owner/repo' from any github.com URL in project_urls."""
    for url in project_urls.values():
        if url and "github.com/" in url:
            parts = url.split("github.com/")[1].rstrip("/").split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    return None


def get_release_notes_from_github() -> str:
    """Get the latest release notes for MLflow from Github"""
    info = get_package_info("mlflow", max_versions=30)
    return info


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts using the Gemini embedding model."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_index(document_store: dict[str, str]) -> list[dict]:
    """Embed all documents and return an index of {id, text, embedding}."""
    doc_ids = list(document_store.keys())
    doc_texts = list(document_store.values())
    embeddings = get_embeddings(doc_texts)

    index = []
    for doc_id, text, emb in zip(doc_ids, doc_texts, embeddings):
        index.append({"id": doc_id, "text": text, "embedding": emb})

    return index

INDEX = build_index(DOCUMENT_STORE)


mlflow_docs_retriever = SemanticRetriever(
    index=INDEX,
    embedding_client=openai_client,
    embedding_model=EMBEDDING_MODEL,   # Gemini's embedding model
    top_k=3,
    name="mlflow_docs",
)

agent = MLflowDocsAgent()
agent.add_tools([get_mlflow_version_pypi, get_release_notes_from_github])
agent.add_retriever(mlflow_docs_retriever)
#mlflow.models.set_model(agent)