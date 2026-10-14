#!/usr/bin/env python
"""Derive the journal builds of the manuscript from submission/manuscript.tex.

submission/manuscript.tex is the master (article class; arXiv and Open Journal of
Astrophysics build). It carries comment markers that bracket the parts every venue
shares:

    %% ABSTRACT-START ... %% ABSTRACT-END     abstract text
    %% KEYWORDS-START ... %% KEYWORDS-END     the \\textbf{Keywords:} line
    %% BODY-START     ... %% BODY-END         Introduction through Acknowledgements
    %% APPENDIX-START ... %% APPENDIX-END     appendices

Inside the BODY region, right after the Acknowledgements, an optional pair

    %% DECLARATIONS-START ... %% DECLARATIONS-END

brackets the \\section*{} declarations (competing interests, funding, CRediT, generative
AI). Elsevier wants them directly above the references, so the elsarticle build moves
the block there. RAS style puts Data availability last, so the mnras build emits
Acknowledgements, then the declarations, then Data availability. A missing pair is an
empty block.

This script slices those parts out and wraps them in venue front matter:

    submission/manuscript_elsarticle.tex   Astronomy and Computing (Elsevier, elsarticle)
    submission/manuscript_mnras.tex        RAS Techniques and Instruments (mnras class;
                                           RASTI accepts non-RASTI class files at
                                           submission, and its own class is an mnras
                                           derivative, so swapping the class name is
                                           the only change if the RASTI file is used)

Run after every edit of manuscript.tex:

    python scripts/revision/build_venue_variants.py
    cd submission && latexmk -pdf manuscript_elsarticle.tex && latexmk -pdf manuscript_mnras.tex

The convenience macros and \\input{revision_numbers.tex} are copied verbatim from the
master, so the journal builds quote exactly the same generated numbers.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUB = REPO / "submission"
MASTER = SUB / "manuscript.tex"

TITLE = (
    "A masked-line ablation audit of a LightGBM F/G/K temperature-class classifier for "
    "Gaia-ESO UVES spectra: what survives, and a train--test scale mismatch"
)
SHORT_TITLE = "Masked-line ablation audit of a Gaia-ESO UVES classifier"
AUTHOR = "Tirthesh Jani"
ORCID = "0009-0005-5965-4409"
EMAIL = "tirtheshjani@gmail.com"
AFFIL = "Lakehead University, Barrie, Ontario, Canada"


def _slice(text: str, name: str) -> str:
    m = re.search(rf"%% {name}-START\n(.*?)\n%% {name}-END", text, re.S)
    if m is None:
        raise SystemExit(f"marker pair {name} not found in {MASTER}")
    return m.group(1)


def _end_matter(body: str) -> tuple[str, str, str, str]:
    """Split the BODY slice into (main text, Data availability, Acknowledgements, declarations).

    The master has Data availability before Acknowledgements and the optional
    DECLARATIONS block after them; the venues reorder these pieces.
    """
    decl = ""
    m = re.search(r"%% DECLARATIONS-START\n(.*?)\n%% DECLARATIONS-END", body, re.S)
    if m is not None:
        decl = m.group(1)
        body = body[: m.start()] + body[m.end() :]
    data_head = "\\section*{Data availability}"
    ack_head = "\\section*{Acknowledgements}"
    i_data, i_ack = body.index(data_head), body.index(ack_head)
    if i_ack < i_data:
        raise SystemExit("expected Data availability before Acknowledgements in the master")
    return body[:i_data], body[i_data:i_ack], body[i_ack:], decl


def _join(*parts: str) -> str:
    """Join non-empty parts with one blank line."""
    return "\n\n".join(p.strip("\n") for p in parts if p.strip("\n"))


def _pdf_metadata(keywords: list[str]) -> str:
    title = TITLE.replace("--", "-")
    kw = "; ".join(keywords)
    return f"\\hypersetup{{pdftitle={{{title}}}, pdfauthor={{{AUTHOR}}}, pdfkeywords={{{kw}}}}}"


def _macros(text: str) -> str:
    start = text.index("% Convenience macros")
    end = text.index("\\input{revision_numbers.tex}") + len("\\input{revision_numbers.tex}")
    return text[start:end]


def _abstract(text: str) -> str:
    return _slice(text, "ABSTRACT").replace("\\noindent ", "", 1)


def _keywords(text: str) -> list[str]:
    line = _slice(text, "KEYWORDS")
    line = line.replace("\\noindent \\textbf{Keywords:}", "").strip()
    return [k.strip() for k in line.split(";") if k.strip()]


def _strip_float_placement(text: str) -> str:
    """Drop [ht]-style placement options; the mnras class prints them as text."""
    return re.sub(r"\\begin\{(table|figure)(\*?)\}\[[a-zA-Z!]*\]", r"\\begin{\1\2}", text)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(REPO)}")


def build_elsarticle(master: str) -> str:
    keywords = _keywords(master)
    kw = " \\sep ".join(keywords)
    abstract = _abstract(master)
    main, data, ack, decl = _end_matter(_slice(master, "BODY"))
    # Elsevier: declarations directly above the references.
    body = _join(main, data, ack, decl)
    return f"""% Astronomy and Computing build. GENERATED by scripts/revision/build_venue_variants.py
% from submission/manuscript.tex; do not edit by hand.
\\documentclass[preprint,12pt,authoryear]{{elsarticle}}

\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{microtype}}
\\usepackage{{xspace}}
\\usepackage[hidelinks]{{hyperref}}
{_pdf_metadata(keywords)}
\\usepackage{{xurl}}
\\Urlmuskip=0mu plus 1mu\\relax
\\setlength{{\\emergencystretch}}{{1.5em}}

\\journal{{Astronomy and Computing}}

{_macros(master)}

\\begin{{document}}

\\begin{{frontmatter}}

\\title{{{TITLE}}}

\\author[lu]{{{AUTHOR}\\corref{{cor1}}}}
\\ead{{{EMAIL}}}
\\cortext[cor1]{{Corresponding author. ORCID: \\href{{https://orcid.org/{ORCID}}}{{{ORCID}}}}}
\\affiliation[lu]{{organization={{Lakehead University}}, city={{Barrie}}, state={{Ontario}}, country={{Canada}}}}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{keyword}}
{kw}
\\end{{keyword}}

\\end{{frontmatter}}

{body}

\\bibliographystyle{{elsarticle-harv}}
\\bibliography{{manuscript_citations}}

\\appendix
{_slice(master, "APPENDIX")}

\\end{{document}}
"""


def build_mnras(master: str) -> str:
    keywords = _keywords(master)
    kw = " -- ".join(keywords)
    abstract = _abstract(master)
    main, data, ack, decl = _end_matter(_slice(master, "BODY"))
    # RAS style: Acknowledgements, declarations, then Data availability last.
    body = _strip_float_placement(_join(main, ack, decl, data))
    return f"""% RAS Techniques and Instruments build. GENERATED by scripts/revision/build_venue_variants.py
% from submission/manuscript.tex; do not edit by hand.
% RASTI accepts non-RASTI class files at submission. Its own class is an mnras derivative;
% if the rasti class file is available, replace {{mnras}} below with {{rasti}}.
\\documentclass[fleqn,usenatbib]{{mnras}}

\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{xspace}}
\\hypersetup{{hidelinks}}  % mnras loads hyperref itself
{_pdf_metadata(keywords)}
\\usepackage{{xurl}}
\\setcounter{{secnumdepth}}{{3}}  % keep \\paragraph headings unnumbered
\\date{{Accepted XXX. Received YYY; in original form ZZZ}}
\\Urlmuskip=0mu plus 1mu\\relax
\\setlength{{\\emergencystretch}}{{1.5em}}

{_macros(master)}
\\renewcommand{{\\appendixbreak}}{{\\relax}}  % floats flow in two columns; no forced break before the appendix figures

\\title[{SHORT_TITLE}]{{{TITLE}}}

\\author[T. Jani]{{{AUTHOR}$^{{1}}$\\thanks{{E-mail: {EMAIL}; ORCID: {ORCID}}}
\\\\
$^{{1}}${AFFIL}
}}

\\begin{{document}}
\\label{{firstpage}}
\\pagerange{{\\pageref{{firstpage}}--\\pageref{{lastpage}}}}
\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{keywords}}
{kw}
\\end{{keywords}}

{body}

\\bibliographystyle{{mnras}}
\\bibliography{{manuscript_citations}}

\\appendix
{_strip_float_placement(_slice(master, "APPENDIX"))}

\\clearpage
\\bsp
\\label{{lastpage}}
\\end{{document}}
"""


def main() -> None:
    master = MASTER.read_text(encoding="utf-8")
    _write(SUB / "manuscript_elsarticle.tex", build_elsarticle(master))
    _write(SUB / "manuscript_mnras.tex", build_mnras(master))


if __name__ == "__main__":
    main()
