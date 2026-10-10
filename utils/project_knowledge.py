"""Concise, source-grounded theory answers about the SupplyGuard project."""

import ast
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def _clean_markdown(lines: List[str]) -> str:
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(not cell or set(cell) <= {"-", ":"} for cell in cells):
                continue
            if cells and cells[0].lower() in {"module", "feature", "name"}:
                continue
            line = ": ".join(cell for cell in cells if cell)
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"[*_`]+", "", line)
        cleaned.append(line)
    return " ".join(cleaned)


def _short_theory(text: str, max_length: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."


def _add_chunk(chunks: List[Dict[str, object]], source: str,
               line: int, text: str, context: str = "") -> None:
    content = " ".join(part for part in (context, text) if part).strip()
    if len(content) < 24:
        return
    search_path = source.replace("/", " ").replace("_", " ").replace(".py", "")
    chunks.append({
        "source": source,
        "line": line,
        "content": _short_theory(content),
        "search_text": f"{search_path} {content}",
    })


def _markdown_chunks(source: str, lines: List[str], chunks: List[Dict[str, object]]) -> None:
    heading = ""
    paragraph: List[str] = []
    paragraph_start = 1

    def flush(start_line: int) -> None:
        text = _clean_markdown(paragraph)
        if text:
            _add_chunk(chunks, source, start_line, text, heading)
        paragraph.clear()

    for line_number, line in enumerate(lines, start=1):
        if line.startswith("#"):
            flush(paragraph_start)
            heading = re.sub(r"^#{1,6}\s*", "", line).strip()
            paragraph_start = line_number + 1
        elif line.startswith("|"):
            flush(paragraph_start)
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            table_text = _clean_markdown([line])
            if heading.lower() == "features" and len(cells) >= 2 and cells[0] == "Demand Forecasting":
                table_text = f"The Demand Forecasting module compares these regression models: {cells[1]}."
            _add_chunk(chunks, source, line_number, table_text, heading)
            paragraph_start = line_number + 1
        elif line.strip():
            if not paragraph:
                paragraph_start = line_number
            paragraph.append(line)
        else:
            flush(paragraph_start)
            paragraph_start = line_number + 1
    flush(paragraph_start)


def _python_theory_chunks(source: str, text: str, chunks: List[Dict[str, object]]) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return

    module_doc = ast.get_docstring(tree)
    if module_doc:
        _add_chunk(chunks, source, 1, module_doc)

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                _add_chunk(chunks, source, node.lineno, docstring, node.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if getattr(node.func.value, "id", None) == "st" and node.func.attr in {"markdown", "caption", "info"}:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    _markdown_chunks(source, node.args[0].value.splitlines(), chunks)


@lru_cache(maxsize=1)
def _build_project_index() -> Tuple[TfidfVectorizer, object, TfidfVectorizer, object, List[Dict[str, object]]]:
    project_root = Path(__file__).resolve().parents[1]
    source_files = [project_root / "README.md", project_root / "app.py", project_root / "config.py"]

    for directory in ("models", "utils", "pages", "data"):
        source_files.extend(sorted((project_root / directory).glob("*.py")))

    chunks: List[Dict[str, object]] = []
    for source_path in source_files:
        if not source_path.is_file() or source_path == Path(__file__):
            continue
        try:
            lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        relative_path = source_path.relative_to(project_root).as_posix()
        if source_path.suffix.lower() == ".md":
            _markdown_chunks(relative_path, lines, chunks)
        elif source_path.suffix.lower() == ".py":
            _python_theory_chunks(relative_path, "\n".join(lines), chunks)

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b",
        sublinear_tf=True,
    )
    word_matrix = word_vectorizer.fit_transform([chunk["search_text"] for chunk in chunks])
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    char_matrix = char_vectorizer.fit_transform([chunk["search_text"] for chunk in chunks])
    return word_vectorizer, word_matrix, char_vectorizer, char_matrix, chunks


def search_project_knowledge(question: str, limit: int = 2) -> Optional[str]:
    """Return up to two concise theory points with their source files."""
    question = question.strip()
    if len(question) < 3:
        return None

    word_vectorizer, word_matrix, char_vectorizer, char_matrix, chunks = _build_project_index()
    word_query = word_vectorizer.transform([question])
    if word_query.nnz == 0:
        return None
    char_query = char_vectorizer.transform([question])
    word_scores = (word_matrix @ word_query.T).toarray().ravel()
    char_scores = (char_matrix @ char_query.T).toarray().ravel()
    scores = 0.75 * word_scores + 0.25 * char_scores
    if re.search(r"\bmodels?\b|\balgorithms?\b", question.lower()):
        known_models = (
            "linear regression", "random forest", "gradient boosting", "xgboost",
            "logistic regression", "isolation forest",
        )
        for index, chunk in enumerate(chunks):
            content = str(chunk["content"]).lower()
            if any(name in content for name in known_models):
                scores[index] += 0.5
            elif "model" in content or "algorithm" in content:
                scores[index] += 0.1
    ranked_indices = scores.argsort()[::-1]

    selected = []
    source_counts: Dict[str, int] = {}
    for index in ranked_indices:
        if word_scores[index] == 0 or scores[index] < 0.02:
            break
        chunk = chunks[index]
        source = str(chunk["source"])
        if source_counts.get(source, 0) >= 1:
            continue
        selected.append(chunk)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= limit:
            break

    if not selected:
        return None

    points = [f"- {chunk['content']}" for chunk in selected]
    sources = ", ".join(dict.fromkeys(str(chunk["source"]) for chunk in selected))
    points.append(f"Sources: {sources}")
    return "\n".join(points)