"""Guards for the public repository: privacy and unsupported claims."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw data and archived numeric results are excluded: long digit runs there are
# measurements, not identifiers.
SKIP_PREFIXES = ("data/", "legacy/results/", "results/")

# Built by concatenation so this file does not contain the word it bans.
BANNED_CLAIM = re.compile("calib" + "rat", re.IGNORECASE)
STUDENT_ID = re.compile(r"(?<!\d)\d{9}(?!\d)")
USER_PATH = re.compile(r"/Users/\w+|/home/\w+|C:\\Users\\")
PERSONAL_EMAIL = re.compile(r"[\w.+-]+@(gmail|outlook|hotmail|yahoo|qq|163|sjsu)\.\w+")


def tracked_text_files() -> list[tuple[str, str]]:
    names = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    files = []
    for name in names:
        if name.startswith(SKIP_PREFIXES):
            continue
        try:
            files.append((name, (ROOT / name).read_text(encoding="utf-8")))
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary, or deleted in the working tree
    return files


def find(pattern: re.Pattern[str]) -> list[str]:
    hits = []
    for name, text in tracked_text_files():
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{name}:{lineno}")
    return hits


def test_no_student_ids() -> None:
    assert find(STUDENT_ID) == []


def test_no_user_paths_or_personal_emails() -> None:
    assert find(USER_PATH) == []
    assert find(PERSONAL_EMAIL) == []


def test_no_unsupported_fit_claim() -> None:
    # Until the processing-time model is fitted to real measurements (Phase 3),
    # no file may describe the simulator with this word. See docs/decisions.md D4.
    assert find(BANNED_CLAIM) == []


def test_no_pdfs_tracked() -> None:
    # PDFs cannot be scanned by the checks above; the removed course PDFs
    # contained student IDs. See docs/decisions.md D3.
    names = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert [n for n in names if n.lower().endswith(".pdf")] == []


def test_generated_doc_blocks_are_up_to_date() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/render_docs.py", "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
