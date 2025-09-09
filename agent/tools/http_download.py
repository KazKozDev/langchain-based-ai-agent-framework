"""
HTTP download tool for binary-safe file downloads (e.g., PDFs).
Validates PDFs by checking %%EOF and attempting to open with pypdf.
Saves under data/files/ by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, HttpUrl

try:
    import pypdf
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


class HttpDownloadInput(BaseModel):
    url: HttpUrl = Field(description="Direct URL to download (use for binary content like PDFs)")
    filename: Optional[str] = Field(default=None, description="Optional target filename; if omitted, derived from URL")


def _default_files_dir() -> Path:
    project_root = Path(__file__).parent.parent.parent
    files_dir = project_root / "data" / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    return files_dir


def _normalize_target(filename: Optional[str], url: str) -> Path:
    base = _default_files_dir()
    if filename:
        name = Path(filename).name  # sanitize
    else:
        name = Path(url.split("?", 1)[0]).name or "download.bin"
    # forbid absolute
    return base / name


def _validate_pdf(path: Path) -> Tuple[bool, str]:
    try:
        # Quick EOF check
        with open(path, "rb") as f:
            f.seek(-10, os.SEEK_END)
            tail = f.read().decode(errors="ignore")
        if "%%EOF" not in tail:
            return False, "PDF does not end with %%EOF"
        if not PDF_AVAILABLE:
            return True, "pypdf not available; EOF check passed"
        reader = PdfReader(str(path))
        _ = len(reader.pages)
        title = reader.metadata.get("/Title") if reader.metadata else None
        return True, f"PDF ok; pages={_}; title={title}"
    except Exception as e:
        return False, f"PDF validation error: {e}"


def http_download(url: str, filename: Optional[str] = None) -> str:
    target = _normalize_target(filename, url)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        return f"Error: download failed: {e}"

    # Validate PDF if applicable
    if target.suffix.lower() == ".pdf":
        ok, msg = _validate_pdf(target)
        status = "OK" if ok else "FAILED"
        return f"Saved to: {target}\nPDF validation: {status} - {msg}"

    return f"Saved to: {target}"


def get_http_download_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=http_download,
        name="http_download",
        description=(
            "Download a file via HTTP(S) in binary-safe mode. Use this for PDFs and other binaries. "
            "Arguments: url, optional filename. Saved under data/files/."
        ),
        args_schema=HttpDownloadInput,
    )
