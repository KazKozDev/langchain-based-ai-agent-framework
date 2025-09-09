"""
ArXiv tool suite

Provides multiple tools:
- arxiv_metadata: fetch title/abstract from the abs page with robust parsing.
- arxiv_search: query arXiv via official API and return structured results.
- arxiv_versions: parse submission/version history from abs page.
- arxiv_bibtex: fetch BibTeX entry for a paper.
- arxiv_pdf_info: return canonical PDF URL and a suggested filename.

Each tool is exported as a LangChain StructuredTool factory.
"""
from __future__ import annotations

import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET
from langchain_core.tools import StructuredTool


ID_RE = re.compile(r"^(?:(?:\w+(?:-\w+)*/)?\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$")


def _normalize_id(identifier: str) -> Optional[str]:
    """Extract a canonical arXiv ID from an input string (ID or URL)."""
    s = identifier.strip()
    # URL forms
    if s.startswith("http://") or s.startswith("https://"):
        u = urlparse(s)
        parts = [p for p in u.path.split("/") if p]
        # /abs/<id>
        if len(parts) >= 2 and parts[0] == "abs":
            return parts[1]
        # /pdf/<id>.pdf
        if len(parts) >= 2 and parts[0] == "pdf":
            id_pdf = parts[1]
            return id_pdf[:-4] if id_pdf.endswith(".pdf") else id_pdf
        # Some pages may embed id as last segment
        if parts:
            cand = parts[-1]
            if cand.endswith(".pdf"):
                cand = cand[:-4]
            if ID_RE.match(cand):
                return cand
        return None
    # Plain ID
    return s if ID_RE.match(s) else None


def _fetch_abs_html(arxiv_id: str) -> str:
    url = f"https://arxiv.org/abs/{arxiv_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _extract_with_meta(soup: BeautifulSoup) -> dict:
    data = {}
    def meta(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.has_attr("content") else None

    # Title
    title = meta("citation_title")
    if not title:
        h1 = soup.find("h1", class_="title") or soup.find("h1", class_="title mathjax")
        if h1:
            t = h1.get_text(" ", strip=True)
            # Often "Title: <text>"
            title = t.split(":", 1)[1].strip() if ":" in t else t
    if title:
        data["title"] = title

    # Authors
    authors: List[str] = []
    author_tags = soup.find_all("meta", attrs={"name": "citation_author"})
    if author_tags:
        authors = [a.get("content", "").strip() for a in author_tags if a.get("content")]
    if not authors:
        auth_div = soup.find("div", class_="authors")
        if auth_div:
            authors = [a.get_text(strip=True) for a in auth_div.find_all("a")]
    if authors:
        data["authors"] = authors

    # Abstract
    abstract = None
    abs_block = soup.find("blockquote", class_="abstract") or soup.find("blockquote", class_="abstract mathjax")
    if abs_block:
        t = abs_block.get_text(" ", strip=True)
        abstract = t.split(":", 1)[1].strip() if ":" in t else t
    if abstract:
        data["abstract"] = abstract

    # DOI
    doi = meta("citation_doi")
    if not doi:
        a = soup.find("a", href=re.compile(r"^https?://doi\.org/"))
        if a:
            doi = a.get_text(strip=True) or a.get("href")
    if doi:
        data["doi"] = doi

    # PDF URL
    pdf_url = meta("citation_pdf_url")
    if not pdf_url:
        # Fallback constructed
        pdf_link = soup.find("a", href=re.compile(r"/pdf/"))
        if pdf_link and pdf_link.get("href"):
            href = pdf_link.get("href")
            pdf_url = f"https://arxiv.org{href}" if href.startswith("/") else href
    if pdf_url:
        data["pdf_url"] = pdf_url

    # Dates
    published = meta("citation_date") or meta("dc.date")
    if published:
        data["published"] = published

    return data


class ArxivInput(BaseModel):
    identifier: str = Field(description="ArXiv ID (e.g., 2502.11705) or URL to abs/pdf page")


def arxiv_metadata(identifier: str) -> dict:
    arxiv_id = _normalize_id(identifier)
    if not arxiv_id:
        return {"success": False, "error": "Could not parse arXiv identifier", "input": identifier}

    try:
        html = _fetch_abs_html(arxiv_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch abs page: {e}", "id": arxiv_id}

    try:
        soup = BeautifulSoup(html, "html.parser")
        data = _extract_with_meta(soup)
        data.update({
            "id": arxiv_id,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        })
        # Minimal validation
        data["success"] = True
        if "title" not in data:
            data["warning"] = "Title not found"
        if "abstract" not in data:
            data["warning"] = (data.get("warning", "") + "; " if data.get("warning") else "") + "Abstract not found"
        return data
    except Exception as e:
        return {"success": False, "error": f"Failed to parse abs page: {e}", "id": arxiv_id}


def get_arxiv_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=arxiv_metadata,
        name="arxiv_metadata",
        description=(
            "Fetch arXiv metadata from the abs page: returns id, title, abstract, authors, doi, pdf_url, published, abs_url. "
            "Input: arXiv ID or URL. Use this to get authoritative title/abstract before summarizing or RAG ingestion."
        ),
        args_schema=ArxivInput,
    )


# ----------------------------
# Search API
# ----------------------------

class ArxivSearchInput(BaseModel):
    query: str = Field(description="Search query, e.g., 'LLM AND agents'")
    max_results: int = Field(default=10, ge=1, le=50, description="Number of results (1-50)")
    sort_by: str = Field(default="relevance", description="Sort by: relevance|lastUpdatedDate|submittedDate")
    sort_order: str = Field(default="descending", description="Sort order: ascending|descending")


def _parse_atom_entry(entry: ET.Element) -> Dict[str, Any]:
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    get = lambda tag: (entry.find(f'atom:{tag}', ns).text if entry.find(f'atom:{tag}', ns) is not None else None)
    data: Dict[str, Any] = {}
    data['id'] = get('id')
    data['title'] = (get('title') or '').strip()
    data['summary'] = (get('summary') or '').strip()
    data['published'] = get('published')
    data['updated'] = get('updated')
    # authors
    data['authors'] = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns) if a.find('atom:name', ns) is not None]
    # links
    links = entry.findall('atom:link', ns)
    pdf_url = None
    abs_url = None
    for l in links:
        rel = l.get('rel')
        href = l.get('href')
        if l.get('type') == 'application/pdf':
            pdf_url = href
        if rel == 'alternate':
            abs_url = href
    data['pdf_url'] = pdf_url
    data['abs_url'] = abs_url
    # primary_category
    pc = entry.find('arxiv:primary_category', ns)
    if pc is not None:
        data['primary_category'] = pc.get('term')
    # categories
    data['categories'] = [c.get('term') for c in entry.findall('atom:category', ns) if c.get('term')]
    # derive short id
    if data['id'] and data['id'].startswith('http'):
        m = re.search(r'arxiv\.org/abs/([^/]+)$', data['id'])
        if m:
            data['arxiv_id'] = m.group(1)
    return data


def arxiv_search(query: str, max_results: int = 10, sort_by: str = "relevance", sort_order: str = "descending") -> Dict[str, Any]:
    base = "http://export.arxiv.org/api/query"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': sort_order,
    }
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = [
            _parse_atom_entry(e)
            for e in root.findall('atom:entry', ns)
        ]
        return {'success': True, 'count': len(entries), 'results': entries}
    except Exception as e:
        return {'success': False, 'error': f'API error: {e}'}


def get_arxiv_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=arxiv_search,
        name="arxiv_search",
        description=(
            "Search arXiv via official API. Returns structured results including id, title, summary, authors, pdf_url, abs_url, categories."
        ),
        args_schema=ArxivSearchInput,
    )


# ----------------------------
# Versions (submission history)
# ----------------------------

class ArxivVersionsInput(BaseModel):
    identifier: str = Field(description="ArXiv ID or URL")


def arxiv_versions(identifier: str) -> Dict[str, Any]:
    arxiv_id = _normalize_id(identifier)
    if not arxiv_id:
        return {"success": False, "error": "Could not parse arXiv identifier", "input": identifier}
    try:
        html = _fetch_abs_html(arxiv_id)
        soup = BeautifulSoup(html, "html.parser")
        hist = soup.find("div", class_="submission-history")
        versions: List[Dict[str, Any]] = []
        if hist:
            for li in hist.find_all("li"):
                text = li.get_text(" ", strip=True)
                # e.g., [v2] Tue, 3 Jan 2023 12:34:56 UTC (123 KB)
                m = re.match(r"\[(v\d+)\]\s+(.*?)(?:\s*\((.*?)\))?$", text)
                if m:
                    versions.append({
                        'version': m.group(1),
                        'date': m.group(2),
                        'size': m.group(3) or None,
                    })
        return {"success": True, "id": arxiv_id, "versions": versions}
    except Exception as e:
        return {"success": False, "error": f"Failed to parse versions: {e}", "id": arxiv_id}


def get_arxiv_versions_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=arxiv_versions,
        name="arxiv_versions",
        description=("Get arXiv submission/version history from the abs page."),
        args_schema=ArxivVersionsInput,
    )


# ----------------------------
# BibTeX
# ----------------------------

class ArxivBibTeXInput(BaseModel):
    identifier: str = Field(description="ArXiv ID or URL")


def arxiv_bibtex(identifier: str) -> Dict[str, Any]:
    arxiv_id = _normalize_id(identifier)
    if not arxiv_id:
        return {"success": False, "error": "Could not parse arXiv identifier", "input": identifier}
    url = f"https://arxiv.org/bibtex/{arxiv_id}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        bib = r.text.strip()
        return {"success": True, "id": arxiv_id, "bibtex": bib}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch BibTeX: {e}", "id": arxiv_id}


def get_arxiv_bibtex_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=arxiv_bibtex,
        name="arxiv_bibtex",
        description=("Fetch BibTeX citation for an arXiv paper."),
        args_schema=ArxivBibTeXInput,
    )


# ----------------------------
# PDF info (URL + suggested filename)
# ----------------------------

class ArxivPdfInfoInput(BaseModel):
    identifier: str = Field(description="ArXiv ID or URL")


def arxiv_pdf_info(identifier: str) -> Dict[str, Any]:
    arxiv_id = _normalize_id(identifier)
    if not arxiv_id:
        return {"success": False, "error": "Could not parse arXiv identifier", "input": identifier}
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{arxiv_id}.pdf"
    return {"success": True, "id": arxiv_id, "pdf_url": pdf_url, "filename": filename}


def get_arxiv_pdf_info_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=arxiv_pdf_info,
        name="arxiv_pdf_info",
        description=("Return canonical PDF URL and suggested filename for an arXiv paper."),
        args_schema=ArxivPdfInfoInput,
    )
