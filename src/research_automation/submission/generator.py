"""
Paper Generator
===============

Generate complete LaTeX papers from templates and content.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

from .citations import Citation, CitationManager
from .templates import (
    Author,
    Conference,
    PaperMetadata,
    FIGURE_TEMPLATE,
    TABLE_TEMPLATE,
    SECTION_TEMPLATES,
    get_template,
)


@dataclass
class Section:
    """A paper section."""

    name: str
    title: str
    content: str
    subsections: list["Section"] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)


@dataclass
class Paper:
    """A complete paper."""

    metadata: PaperMetadata
    sections: list[Section] = field(default_factory=list)
    citations: CitationManager = field(default_factory=CitationManager)

    def add_section(
        self,
        name: str,
        title: str,
        content: str,
    ) -> Section:
        """Add a section to the paper."""
        section = Section(name=name, title=title, content=content)
        self.sections.append(section)
        return section

    def add_citation(self, citation: Citation) -> str:
        """Add a citation and return its key."""
        return self.citations.add(citation)

    def cite(self, key: str) -> str:
        """Return LaTeX cite command."""
        return f"\\cite{{{key}}}"


class PaperGenerator:
    """Generate LaTeX papers."""

    def __init__(self):
        self.env = Environment(loader=BaseLoader())

    def render_template(self, template: str, context: dict) -> str:
        """Render a Jinja2 template."""
        tmpl = self.env.from_string(template)
        return tmpl.render(**context)

    def generate_figure(
        self,
        path: str,
        caption: str,
        label: str,
        width: str = r"\linewidth",
        placement: str = "t",
    ) -> str:
        """Generate LaTeX figure."""
        return self.render_template(FIGURE_TEMPLATE, {
            "path": path,
            "caption": caption,
            "label": label,
            "width": width,
            "placement": placement,
        })

    def generate_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        caption: str,
        label: str,
        placement: str = "t",
    ) -> str:
        """Generate LaTeX table."""
        columns = "l" + "c" * (len(headers) - 1)
        header_str = " & ".join(headers)
        row_strs = [" & ".join(str(cell) for cell in row) for row in rows]

        return self.render_template(TABLE_TEMPLATE, {
            "columns": columns,
            "header": header_str,
            "rows": row_strs,
            "caption": caption,
            "label": label,
            "placement": placement,
        })

    def generate_section(self, section: Section, level: int = 1) -> str:
        """Generate LaTeX section."""
        cmd = "section" if level == 1 else "subsection" if level == 2 else "subsubsection"
        lines = [f"\\{cmd}{{{section.title}}}", "", section.content, ""]

        # Add figures
        for fig in section.figures:
            lines.append(self.generate_figure(**fig))
            lines.append("")

        # Add tables
        for tab in section.tables:
            lines.append(self.generate_table(**tab))
            lines.append("")

        # Add subsections
        for subsection in section.subsections:
            lines.append(self.generate_section(subsection, level + 1))

        return "\n".join(lines)

    def generate(self, paper: Paper) -> str:
        """Generate complete LaTeX document."""
        # Generate section content
        content_parts = []
        for section in paper.sections:
            content_parts.append(self.generate_section(section))

        content = "\n\n".join(content_parts)

        # Render main template
        template = get_template(paper.metadata.conference)
        context = {
            "title": paper.metadata.title,
            "authors": paper.metadata.authors,
            "abstract": paper.metadata.abstract,
            "keywords": paper.metadata.keywords,
            "content": content,
            "acknowledgments": paper.metadata.acknowledgments,
        }

        return self.render_template(template, context)

    def save(
        self,
        paper: Paper,
        output_dir: str | Path,
        main_file: str = "main.tex",
    ) -> Path:
        """Save paper to directory with all files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main.tex
        main_path = output_dir / main_file
        main_path.write_text(self.generate(paper))

        # Save references.bib
        bib_path = output_dir / "references.bib"
        paper.citations.save_bibtex(bib_path)

        return output_dir

    def compile_pdf(
        self,
        tex_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path | None:
        """Compile LaTeX to PDF using pdflatex."""
        tex_path = Path(tex_path)
        if output_dir is None:
            output_dir = tex_path.parent

        output_dir = Path(output_dir)

        # Check if pdflatex is available
        if shutil.which("pdflatex") is None:
            raise RuntimeError("pdflatex not found. Install TeX Live or MiKTeX.")

        try:
            # Run pdflatex twice for references
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
                    capture_output=True,
                    check=True,
                    cwd=tex_path.parent,
                )

            # Run bibtex
            aux_file = output_dir / tex_path.stem
            subprocess.run(
                ["bibtex", str(aux_file)],
                capture_output=True,
                cwd=output_dir,
            )

            # Run pdflatex again
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
                capture_output=True,
                cwd=tex_path.parent,
            )

            pdf_path = output_dir / f"{tex_path.stem}.pdf"
            if pdf_path.exists():
                return pdf_path

        except subprocess.CalledProcessError:
            pass

        return None


def create_paper(
    title: str,
    authors: list[dict],
    abstract: str,
    conference: str = "generic",
    keywords: list[str] | None = None,
) -> Paper:
    """
    Create a new paper with basic metadata.

    Args:
        title: Paper title
        authors: List of author dicts with name, email, affiliation
        abstract: Paper abstract
        conference: Conference name (neurips, cvpr, etc.)
        keywords: Optional list of keywords

    Returns:
        Paper object ready for content
    """
    author_objs = [
        Author(
            name=a["name"],
            email=a.get("email", ""),
            affiliation=a.get("affiliation", ""),
            equal_contribution=a.get("equal_contribution", False),
            corresponding=a.get("corresponding", False),
        )
        for a in authors
    ]

    try:
        conf = Conference(conference.lower())
    except ValueError:
        conf = Conference.GENERIC

    metadata = PaperMetadata(
        title=title,
        authors=author_objs,
        abstract=abstract,
        keywords=keywords or [],
        conference=conf,
    )

    return Paper(metadata=metadata)


# Predefined section structures
STANDARD_SECTIONS = [
    ("introduction", "Introduction"),
    ("related_work", "Related Work"),
    ("methods", "Methods"),
    ("experiments", "Experiments"),
    ("results", "Results"),
    ("discussion", "Discussion"),
    ("conclusion", "Conclusion"),
]

MEDICAL_SECTIONS = [
    ("introduction", "Introduction"),
    ("related_work", "Related Work"),
    ("materials", "Materials and Methods"),
    ("results", "Results"),
    ("discussion", "Discussion"),
    ("conclusion", "Conclusion"),
]


def create_paper_from_experiment(
    experiment_name: str,
    description: str,
    dataset: str,
    methods: str,
    results: dict,
    metrics: dict,
    authors: list[dict],
    conference: str = "generic",
) -> Paper:
    """Create a paper from experiment results."""
    # Generate abstract
    best_metric = max(metrics.items(), key=lambda x: x[1])
    abstract = f"""
We present {experiment_name}, a method for {description.lower()}.
Using the {dataset} dataset, our approach achieves {best_metric[1]:.1%} {best_metric[0]}.
{methods}
"""

    paper = create_paper(
        title=experiment_name,
        authors=authors,
        abstract=abstract.strip(),
        conference=conference,
    )

    # Add sections
    paper.add_section(
        "introduction",
        "Introduction",
        f"This paper presents {experiment_name}, addressing the problem of {description.lower()}.",
    )

    paper.add_section(
        "methods",
        "Methods",
        f"\\subsection{{Dataset}}\\n{dataset}\\n\\n\\subsection{{Approach}}\\n{methods}",
    )

    # Results section with table
    results_content = "Table~\\ref{tab:results} shows our experimental results."
    paper.add_section("results", "Results", results_content)

    # Add results table
    paper.sections[-1].tables.append({
        "headers": ["Metric", "Value"],
        "rows": [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics.items()],
        "caption": "Experimental results",
        "label": "results",
    })

    paper.add_section(
        "conclusion",
        "Conclusion",
        f"We presented {experiment_name} for {description.lower()}, achieving {best_metric[1]:.1%} {best_metric[0]}.",
    )

    return paper
