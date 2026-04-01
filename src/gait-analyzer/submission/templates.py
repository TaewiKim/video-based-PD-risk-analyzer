"""
Paper Templates for Conference Submissions
==========================================

LaTeX templates for NeurIPS, CVPR, ICML, MICCAI, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Conference(Enum):
    """Supported conference formats."""

    NEURIPS = "neurips"
    CVPR = "cvpr"
    ICCV = "iccv"
    ICML = "icml"
    ICLR = "iclr"
    AAAI = "aaai"
    MICCAI = "miccai"
    GENERIC = "generic"


@dataclass
class Author:
    """Paper author information."""

    name: str
    email: str = ""
    affiliation: str = ""
    orcid: str = ""
    equal_contribution: bool = False
    corresponding: bool = False


@dataclass
class PaperMetadata:
    """Paper metadata for submission."""

    title: str
    authors: list[Author]
    abstract: str
    keywords: list[str] = field(default_factory=list)
    conference: Conference = Conference.GENERIC

    # Optional fields
    acknowledgments: str = ""
    supplementary: bool = False
    code_url: str = ""
    data_url: str = ""


# LaTeX Templates
NEURIPS_TEMPLATE = r"""
\documentclass{article}

% NeurIPS style
\usepackage[final]{neurips_2024}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}

\title{{ "{{title}}" }}

{% for author in authors %}
\author{
  {{ author.name }}{% if author.equal_contribution %}\thanks{Equal contribution}{% endif %}{% if author.corresponding %}\thanks{Corresponding author}{% endif %} \\
  {{ author.affiliation }} \\
  \texttt{{ "{" }}{{ author.email }}{{ "}" }}
}
{% endfor %}

\begin{document}

\maketitle

\begin{abstract}
{{ abstract }}
\end{abstract}

{{ content }}

{% if acknowledgments %}
\section*{Acknowledgments}
{{ acknowledgments }}
{% endif %}

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
"""

CVPR_TEMPLATE = r"""
\documentclass[10pt,twocolumn,letterpaper]{article}

\usepackage{cvpr}
\usepackage{times}
\usepackage{epsfig}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}

\def\cvprPaperID{****}
\def\confName{CVPR}
\def\confYear{2025}

\begin{document}

\title{{ "{{title}}" }}

\author{
{% for author in authors %}
{{ author.name }}{% if not loop.last %} \and {% endif %}
{% endfor %}
}

\maketitle

\begin{abstract}
{{ abstract }}
\end{abstract}

{{ content }}

{\small
\bibliographystyle{ieee_fullname}
\bibliography{references}
}

\end{document}
"""

MICCAI_TEMPLATE = r"""
\documentclass[runningheads]{llncs}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}

\begin{document}

\title{{ "{{title}}" }}

\author{
{% for author in authors %}
{{ author.name }}\inst{{ "{" }}{{ loop.index }}{{ "}" }}{% if not loop.last %} \and {% endif %}
{% endfor %}
}

\institute{
{% for author in authors %}
{{ author.affiliation }}{% if not loop.last %} \and {% endif %}
{% endfor %}
}

\maketitle

\begin{abstract}
{{ abstract }}
\keywords{{ "{" }}{{ keywords | join(", ") }}{{ "}" }}
\end{abstract}

{{ content }}

\bibliographystyle{splncs04}
\bibliography{references}

\end{document}
"""

GENERIC_TEMPLATE = r"""
\documentclass[11pt]{article}

\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}

\title{{ "{{title}}" }}

\author{
{% for author in authors %}
{{ author.name }}{% if author.affiliation %} \\ {{ author.affiliation }}{% endif %}{% if not loop.last %} \and {% endif %}
{% endfor %}
}

\date{\today}

\begin{document}

\maketitle

\begin{abstract}
{{ abstract }}
\end{abstract}

{{ content }}

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""


TEMPLATES = {
    Conference.NEURIPS: NEURIPS_TEMPLATE,
    Conference.CVPR: CVPR_TEMPLATE,
    Conference.ICCV: CVPR_TEMPLATE,  # Same format
    Conference.ICML: NEURIPS_TEMPLATE,  # Similar format
    Conference.ICLR: NEURIPS_TEMPLATE,
    Conference.MICCAI: MICCAI_TEMPLATE,
    Conference.GENERIC: GENERIC_TEMPLATE,
}


def get_template(conference: Conference) -> str:
    """Get LaTeX template for conference."""
    return TEMPLATES.get(conference, GENERIC_TEMPLATE)


# Section templates
SECTION_TEMPLATES = {
    "introduction": r"""
\section{Introduction}

{{ content }}
""",
    "related_work": r"""
\section{Related Work}

{{ content }}
""",
    "methods": r"""
\section{Methods}

{{ content }}
""",
    "experiments": r"""
\section{Experiments}

{{ content }}
""",
    "results": r"""
\section{Results}

{{ content }}
""",
    "discussion": r"""
\section{Discussion}

{{ content }}
""",
    "conclusion": r"""
\section{Conclusion}

{{ content }}
""",
}


# Figure templates
FIGURE_TEMPLATE = r"""
\begin{figure}[{{ placement | default('t') }}]
    \centering
    \includegraphics[width={{ width | default('\\linewidth') }}]{{ "{" }}{{ path }}{{ "}" }}
    \caption{{ "{" }}{{ caption }}{{ "}" }}
    \label{{ "{" }}fig:{{ label }}{{ "}" }}
\end{figure}
"""

TABLE_TEMPLATE = r"""
\begin{table}[{{ placement | default('t') }}]
    \centering
    \caption{{ "{" }}{{ caption }}{{ "}" }}
    \label{{ "{" }}tab:{{ label }}{{ "}" }}
    \begin{tabular}{{ "{" }}{{ columns }}{{ "}" }}
        \toprule
        {{ header }} \\
        \midrule
        {% for row in rows %}
        {{ row }} \\
        {% endfor %}
        \bottomrule
    \end{tabular}
\end{table}
"""
