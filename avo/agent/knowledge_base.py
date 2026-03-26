"""Knowledge base loader and retriever for AVO.

Implements K from the paper -- a directory of domain-specific documents
(CUDA programming guides, PTX ISA docs, reference implementations, etc.)
that the agent can consult during variation steps.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".cu", ".cuh", ".h", ".cpp", ".c", ".rst", ".json"}


@dataclass
class Document:
    """A document in the knowledge base."""

    name: str
    path: str
    content: str
    extension: str

    @property
    def summary(self) -> str:
        lines = self.content.strip().splitlines()
        preview = "\n".join(lines[:5])
        return f"[{self.name}] ({len(lines)} lines)\n{preview}..."


@dataclass
class KnowledgeBase:
    """Domain-specific knowledge base K.

    Loads all supported files from a directory and provides
    keyword-based retrieval for the agent to query.
    """

    documents: list[Document] = field(default_factory=list)
    _index: dict[str, list[int]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_directory(cls, directory: Path | str) -> KnowledgeBase:
        """Load all supported documents from a directory tree."""
        directory = Path(directory)
        if not directory.exists():
            logger.warning("Knowledge base directory does not exist: %s", directory)
            return cls()

        kb = cls()
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:
                try:
                    content = path.read_text(errors="replace")
                    doc = Document(
                        name=path.name,
                        path=str(path.relative_to(directory)),
                        content=content,
                        extension=path.suffix,
                    )
                    kb.documents.append(doc)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)

        kb._build_index()
        logger.info("Loaded knowledge base: %d documents from %s", len(kb.documents), directory)
        return kb

    def _build_index(self) -> None:
        """Build a simple inverted index on lowercased words."""
        self._index.clear()
        for idx, doc in enumerate(self.documents):
            words = set(re.findall(r"\w+", doc.content.lower()))
            words.add(doc.name.lower())
            for w in words:
                self._index.setdefault(w, []).append(idx)

    def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Keyword search across all documents.

        Returns documents ranked by number of query terms matched.
        """
        terms = re.findall(r"\w+", query.lower())
        if not terms:
            return []

        scores: dict[int, int] = {}
        for term in terms:
            for idx in self._index.get(term, []):
                scores[idx] = scores.get(idx, 0) + 1

        ranked = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:max_results]]

    def get_document(self, name: str) -> Document | None:
        """Retrieve a specific document by name or path."""
        for doc in self.documents:
            if doc.name == name or doc.path == name:
                return doc
        return None

    def list_documents(self) -> list[str]:
        """Return all document names."""
        return [doc.path for doc in self.documents]

    def catalog(self) -> str:
        """Return a formatted catalog for the agent's system prompt."""
        if not self.documents:
            return "Knowledge base is empty."
        lines = ["Available knowledge base documents:"]
        for doc in self.documents:
            n_lines = len(doc.content.splitlines())
            lines.append(f"  - {doc.path} ({n_lines} lines, {doc.extension})")
        return "\n".join(lines)
