import ast
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import mlflow
from mlflow.entities import SpanType
import requests

mlflow.langchain.autolog()

MODEL = "gemini-2.5-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0.0,
)

prompt_version = mlflow.genai.load_prompt("prompts:/mlflow-agent-system@prod")
SYSTEM_PROMPT = prompt_version.template

@mlflow.trace(span_type=SpanType.TOOL)
def get_mlflow_version_pypi() -> str:
    """
    Fetches the current stable release of MLflow directly from PyPI.
    Returns a string like "2.8.0" or an error message if the request fails.
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

@mlflow.trace(span_type=SpanType.TOOL)
def get_release_notes_from_github() -> str:
    """Get the latest release notes for MLflow from Github"""
    info = get_package_info("mlflow", max_versions=30)
    return info



@mlflow.trace(span_type=SpanType.TOOL)
def validate_python_syntax(code: str) -> dict:
    """
    Validates Python syntax using the built-in ast module.
    No external dependencies required.

    Args:
        code: A string of Python code to validate.

    Returns:
        A dict with keys: valid, error, line, offset
    """
    try:
        ast.parse(code)
        return {
            "valid": True,
            "error": None,
            "line": None,
            "offset": None,
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e.msg),
            "line": e.lineno,
            "offset": e.offset,
        }
    

tools=[get_mlflow_version_pypi, get_release_notes_from_github, validate_python_syntax]

agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)