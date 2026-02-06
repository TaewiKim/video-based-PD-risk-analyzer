"""Paper submission and formatting utilities."""

from .citations import (
    Citation,
    CitationManager,
    EntryType,
    create_conference_citation,
)
from .generator import (
    Paper,
    PaperGenerator,
    Section,
    create_paper,
    create_paper_from_experiment,
)
from .templates import (
    Author,
    Conference,
    PaperMetadata,
    get_template,
)

__all__ = [
    # Citations
    "Citation",
    "CitationManager",
    "EntryType",
    "create_conference_citation",
    # Generator
    "Paper",
    "PaperGenerator",
    "Section",
    "create_paper",
    "create_paper_from_experiment",
    # Templates
    "Author",
    "Conference",
    "PaperMetadata",
    "get_template",
]
