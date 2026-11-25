"""Shared paths and helpers for every pipeline stage."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "04-Data" / "raw"
CLEAN = ROOT / "04-Data" / "clean"
RESULTS = ROOT / "06-Results"
METRICS = RESULTS / "metrics"
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"
for d in (CLEAN, METRICS, TABLES, FIGS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 395
rng = np.random.default_rng(SEED)


def read_cms_csv(name: str, **kw) -> pd.DataFrame:
    """CMS files are a mix of UTF-8 and Latin-1 (issue I-10); read as strings."""
    path = RAW / name
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, low_memory=False, encoding="latin-1", **kw)


def ccn6(s: pd.Series) -> pd.Series:
    """Normalize a CMS Certification Number to 6 characters (leading zeros kept)."""
    s = s.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    return s.where(s.str.len() >= 6, s.str.zfill(6))


def zip5(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{5})", expand=False)


def phone10(s: pd.Series) -> pd.Series:
    d = s.astype(str).str.replace(r"\D", "", regex=True)
    d = d.where(~d.str.startswith("1") | (d.str.len() != 11), d.str[1:])
    return d.where(d.str.len() == 10)


# Name normalization: expand common abbreviations, drop legal suffixes.
_ABBREV = {
    r"\bST\b": "SAINT", r"\bMT\b": "MOUNT", r"\bMED\b": "MEDICAL", r"\bCTR\b": "CENTER",
    r"\bCNTR\b": "CENTER", r"\bHOSP\b": "HOSPITAL", r"\bREG\b": "REGIONAL",
    r"\bRGNL\b": "REGIONAL", r"\bMEM\b": "MEMORIAL", r"\bMEML\b": "MEMORIAL",
    r"\bCOMM\b": "COMMUNITY", r"\bCMNTY\b": "COMMUNITY", r"\bUNIV\b": "UNIVERSITY",
    r"\bHLTH\b": "HEALTH", r"\bSYS\b": "SYSTEM", r"\bGEN\b": "GENERAL", r"\bCHILDRENS\b": "CHILDREN",
    r"\bCHILDREN'S\b": "CHILDREN", r"&": " AND ",
}
_DROP = r"\b(INC|LLC|L L C|LP|LLP|CORP|CORPORATION|CO|COMPANY|PC|PLLC|THE|OF|DBA)\b"


def norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.upper()
    s = s.str.replace(r"[’']", "", regex=True)
    for pat, rep in _ABBREV.items():
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(r"[^A-Z0-9 ]", " ", regex=True)
    s = s.str.replace(_DROP, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


_STREET = {r"\bSTREET\b": "ST", r"\bAVENUE\b": "AVE", r"\bROAD\b": "RD", r"\bDRIVE\b": "DR",
           r"\bBOULEVARD\b": "BLVD", r"\bHIGHWAY\b": "HWY", r"\bNORTH\b": "N", r"\bSOUTH\b": "S",
           r"\bEAST\b": "E", r"\bWEST\b": "W", r"\bSUITE\b": "STE", r"\bLANE\b": "LN",
           r"\bPARKWAY\b": "PKWY", r"\bPLACE\b": "PL", r"\bCOURT\b": "CT"}


def norm_addr(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True)
    for pat, rep in _STREET.items():
        s = s.str.replace(pat, rep, regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def save_metrics(name: str, obj: dict) -> None:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (METRICS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=conv))


def load_metrics(name: str) -> dict:
    return json.loads((METRICS / f"{name}.json").read_text())


def log(msg: str) -> None:
    print(msg, flush=True)


_TOKEN = re.compile(r"[A-Z0-9]+")
STOP_TOKENS = {"HEALTH", "SYSTEM", "HOSPITAL", "MEDICAL", "CENTER", "REGIONAL", "COMMUNITY",
               "SAINT", "MEMORIAL", "GENERAL", "CARE", "HEALTHCARE", "SERVICES", "NETWORK",
               "AND", "UNIVERSITY", "COUNTY", "CHILDREN", "INCORPORATED", "PARTNERS", "GROUP"}


def key_tokens(name: str) -> set[str]:
    """Distinctive tokens of a normalized name (generic words removed)."""
    return {t for t in _TOKEN.findall(name or "") if t not in STOP_TOKENS and len(t) > 2}
