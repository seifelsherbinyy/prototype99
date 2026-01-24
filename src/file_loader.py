"""
File loader utilities with format registry and parsing support.

Supports CSV/XLSX/XLS/XLSM/JSON with delimiter sniffing and sheet selection.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

ALLOWED_SUFFIXES = (".csv", ".xlsx", ".xls", ".xlsm", ".json")
CSV_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
CSV_DELIMITERS = [",", ";", "\t", "|"]


def _read_sample(path: Path, encoding: str, sample_size: int = 8192) -> str | None:
    try:
        with path.open("r", encoding=encoding, errors="replace") as handle:
            return handle.read(sample_size)
    except Exception as exc:
        logger.debug(f"Failed to read sample for {path} ({encoding}): {exc}")
        return None


def sniff_csv_delimiter(path: Path, encoding: str) -> str | None:
    sample = _read_sample(path, encoding)
    if not sample:
        return None
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS)
        return dialect.delimiter
    except Exception:
        return None


def load_csv(
    path: Path,
    *,
    delimiter: str | None = None,
    encoding: str | None = None,
    encodings: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    encodings_to_try = [encoding] if encoding else (encodings or CSV_ENCODINGS)
    last_error: Exception | None = None
    for candidate in encodings_to_try:
        try:
            sep = delimiter or sniff_csv_delimiter(path, candidate) or ","
            return pd.read_csv(path, encoding=candidate, sep=sep, low_memory=False, **kwargs)
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError(f"Failed to load CSV: {path}") from last_error


def load_excel(
    path: Path,
    *,
    sheet_name: str | int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    engine = "openpyxl" if suffix in (".xlsx", ".xlsm") else None
    return pd.read_excel(path, engine=engine, sheet_name=sheet_name, **kwargs)


def load_json(path: Path, *, lines: bool | None = None, **kwargs: Any) -> pd.DataFrame:
    errors: list[Exception] = []
    for candidate in ([lines] if lines is not None else [True, False]):
        try:
            data = pd.read_json(path, lines=candidate, **kwargs)
            if isinstance(data, pd.DataFrame):
                return data
            return pd.DataFrame(data)
        except Exception as exc:
            errors.append(exc)
            continue
    raise ValueError(f"Failed to load JSON: {path}") from errors[-1] if errors else None


def load_file(
    path: Path | str,
    *,
    delimiter: str | None = None,
    sheet_name: str | int | None = None,
    encoding: str | None = None,
    json_lines: bool | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return load_csv(file_path, delimiter=delimiter, encoding=encoding, **kwargs)
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return load_excel(file_path, sheet_name=sheet_name, **kwargs)
    if suffix == ".json":
        kwargs.pop("nrows", None)
        return load_json(file_path, lines=json_lines, **kwargs)

    raise ValueError(f"Unsupported file format: {suffix}")
