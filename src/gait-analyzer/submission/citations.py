"""
Citation Management
===================

Manage references, BibTeX generation, and citation formatting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


class EntryType(Enum):
    """BibTeX entry types."""

    ARTICLE = "article"
    INPROCEEDINGS = "inproceedings"
    BOOK = "book"
    INCOLLECTION = "incollection"
    MISC = "misc"
    PHDTHESIS = "phdthesis"
    MASTERSTHESIS = "mastersthesis"
    TECHREPORT = "techreport"
    UNPUBLISHED = "unpublished"


@dataclass
class Citation:
    """A citation entry."""

    key: str
    entry_type: EntryType
    title: str
    authors: list[str]
    year: int

    # Optional fields
    journal: str = ""
    booktitle: str = ""
    volume: str = ""
    number: str = ""
    pages: str = ""
    publisher: str = ""
    doi: str = ""
    url: str = ""
    arxiv_id: str = ""
    note: str = ""

    def to_bibtex(self) -> str:
        """Convert to BibTeX format."""
        lines = [f"@{self.entry_type.value}{{{self.key},"]

        # Authors
        author_str = " and ".join(self.authors)
        lines.append(f'  author = {{{author_str}}},')

        # Title
        lines.append(f'  title = {{{self.title}}},')

        # Year
        lines.append(f'  year = {{{self.year}}},')

        # Optional fields
        if self.journal:
            lines.append(f'  journal = {{{self.journal}}},')
        if self.booktitle:
            lines.append(f'  booktitle = {{{self.booktitle}}},')
        if self.volume:
            lines.append(f'  volume = {{{self.volume}}},')
        if self.number:
            lines.append(f'  number = {{{self.number}}},')
        if self.pages:
            lines.append(f'  pages = {{{self.pages}}},')
        if self.publisher:
            lines.append(f'  publisher = {{{self.publisher}}},')
        if self.doi:
            lines.append(f'  doi = {{{self.doi}}},')
        if self.url:
            lines.append(f'  url = {{{self.url}}},')
        if self.arxiv_id:
            lines.append(f'  eprint = {{{self.arxiv_id}}},')
            lines.append(f'  archiveprefix = {{arXiv}},')
        if self.note:
            lines.append(f'  note = {{{self.note}}},')

        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def from_doi(cls, doi: str) -> "Citation | None":
        """Create citation from DOI using CrossRef."""
        try:
            url = f"https://api.crossref.org/works/{doi}"
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()["message"]

            # Parse authors
            authors = []
            for author in data.get("author", []):
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                if name:
                    authors.append(name)

            # Generate key
            first_author = authors[0].split()[-1] if authors else "unknown"
            year = data.get("published-print", data.get("published-online", {}))
            year = year.get("date-parts", [[None]])[0][0] if year else datetime.now().year
            key = f"{first_author.lower()}{year}"

            # Determine entry type
            entry_type = EntryType.ARTICLE
            if "proceedings" in data.get("type", "").lower():
                entry_type = EntryType.INPROCEEDINGS

            return cls(
                key=key,
                entry_type=entry_type,
                title=data.get("title", [""])[0],
                authors=authors,
                year=year,
                journal=data.get("container-title", [""])[0],
                volume=data.get("volume", ""),
                number=data.get("issue", ""),
                pages=data.get("page", ""),
                publisher=data.get("publisher", ""),
                doi=doi,
            )
        except Exception:
            return None

    @classmethod
    def from_arxiv(cls, arxiv_id: str) -> "Citation | None":
        """Create citation from arXiv ID."""
        try:
            # Clean arXiv ID
            arxiv_id = arxiv_id.replace("arXiv:", "").strip()

            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
                response.raise_for_status()

            # Parse XML response
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entry = root.find("atom:entry", ns)
            if entry is None:
                return None

            title = entry.find("atom:title", ns)
            title = title.text.strip().replace("\n", " ") if title is not None else ""

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)

            published = entry.find("atom:published", ns)
            year = int(published.text[:4]) if published is not None else datetime.now().year

            first_author = authors[0].split()[-1] if authors else "unknown"
            key = f"{first_author.lower()}{year}arxiv"

            return cls(
                key=key,
                entry_type=EntryType.MISC,
                title=title,
                authors=authors,
                year=year,
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        except Exception:
            return None


class CitationManager:
    """Manage a collection of citations."""

    def __init__(self):
        self.citations: dict[str, Citation] = {}

    def add(self, citation: Citation) -> str:
        """Add a citation, returning its key."""
        # Handle duplicate keys
        key = citation.key
        if key in self.citations:
            i = 1
            while f"{key}{chr(ord('a') + i - 1)}" in self.citations:
                i += 1
            key = f"{citation.key}{chr(ord('a') + i - 1)}"
            citation.key = key

        self.citations[key] = citation
        return key

    def add_from_doi(self, doi: str) -> str | None:
        """Add citation from DOI."""
        citation = Citation.from_doi(doi)
        if citation:
            return self.add(citation)
        return None

    def add_from_arxiv(self, arxiv_id: str) -> str | None:
        """Add citation from arXiv ID."""
        citation = Citation.from_arxiv(arxiv_id)
        if citation:
            return self.add(citation)
        return None

    def get(self, key: str) -> Citation | None:
        """Get citation by key."""
        return self.citations.get(key)

    def remove(self, key: str) -> bool:
        """Remove citation by key."""
        if key in self.citations:
            del self.citations[key]
            return True
        return False

    def to_bibtex(self) -> str:
        """Export all citations to BibTeX format."""
        entries = [c.to_bibtex() for c in self.citations.values()]
        return "\n\n".join(entries)

    def save_bibtex(self, path: str | Path) -> None:
        """Save citations to BibTeX file."""
        Path(path).write_text(self.to_bibtex())

    def load_bibtex(self, path: str | Path) -> int:
        """Load citations from BibTeX file. Returns count of loaded entries."""
        content = Path(path).read_text()
        return self.parse_bibtex(content)

    def parse_bibtex(self, content: str) -> int:
        """Parse BibTeX content and add citations. Returns count."""
        # Simple BibTeX parser
        pattern = r'@(\w+)\{([^,]+),\s*(.*?)\n\}'
        matches = re.findall(pattern, content, re.DOTALL)

        count = 0
        for entry_type, key, fields_str in matches:
            try:
                # Parse fields
                fields = {}
                for match in re.finditer(r'(\w+)\s*=\s*\{([^}]*)\}', fields_str):
                    fields[match.group(1).lower()] = match.group(2)

                # Create citation
                authors = fields.get("author", "").split(" and ")
                year = int(fields.get("year", datetime.now().year))

                citation = Citation(
                    key=key.strip(),
                    entry_type=EntryType(entry_type.lower()),
                    title=fields.get("title", ""),
                    authors=[a.strip() for a in authors if a.strip()],
                    year=year,
                    journal=fields.get("journal", ""),
                    booktitle=fields.get("booktitle", ""),
                    volume=fields.get("volume", ""),
                    number=fields.get("number", ""),
                    pages=fields.get("pages", ""),
                    publisher=fields.get("publisher", ""),
                    doi=fields.get("doi", ""),
                    url=fields.get("url", ""),
                    arxiv_id=fields.get("eprint", ""),
                )
                self.citations[citation.key] = citation
                count += 1
            except Exception:
                continue

        return count

    def search(self, query: str) -> list[Citation]:
        """Search citations by title or author."""
        query = query.lower()
        results = []
        for citation in self.citations.values():
            if query in citation.title.lower():
                results.append(citation)
            elif any(query in author.lower() for author in citation.authors):
                results.append(citation)
        return results

    def __len__(self) -> int:
        return len(self.citations)

    def __iter__(self):
        return iter(self.citations.values())


# Common conference proceedings
CONFERENCE_BOOKTITLES = {
    "neurips": "Advances in Neural Information Processing Systems",
    "cvpr": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "iccv": "Proceedings of the IEEE/CVF International Conference on Computer Vision",
    "eccv": "European Conference on Computer Vision",
    "icml": "Proceedings of the International Conference on Machine Learning",
    "iclr": "International Conference on Learning Representations",
    "aaai": "Proceedings of the AAAI Conference on Artificial Intelligence",
    "miccai": "Medical Image Computing and Computer Assisted Intervention",
    "acl": "Proceedings of the Annual Meeting of the Association for Computational Linguistics",
    "emnlp": "Proceedings of the Conference on Empirical Methods in Natural Language Processing",
}


def create_conference_citation(
    key: str,
    title: str,
    authors: list[str],
    year: int,
    conference: str,
    pages: str = "",
) -> Citation:
    """Create a conference paper citation."""
    booktitle = CONFERENCE_BOOKTITLES.get(conference.lower(), conference)
    return Citation(
        key=key,
        entry_type=EntryType.INPROCEEDINGS,
        title=title,
        authors=authors,
        year=year,
        booktitle=booktitle,
        pages=pages,
    )
