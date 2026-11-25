"""Stage 00: download every public source into 04-Data/raw and write a manifest.

Re-runnable: files that already exist with a non-zero size are skipped.
Writes 04-Data/raw/_manifest.json (url, bytes, sha256, fetched_at) so the
report can cite exact releases.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "04-Data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "research-download (python-requests)"}
AHRQ = "https://www.ahrq.gov/sites/default/files/wysiwyg/chsp/compendium/"

SOURCES = {
    # --- ground truth: AHRQ Compendium of U.S. Health Systems ---
    "ahrq_hospital_linkage_2022.csv": AHRQ + "chsp-hospital-linkage-2022-rev.csv",
    "ahrq_systems_2022.csv": AHRQ + "chsp-compendium-2022.csv",
    "ahrq_hospital_linkage_2018.csv": AHRQ + "chsp-hospital-linkage-2018.csv",
    "ahrq_systems_2018.csv": AHRQ + "chsp-compendium-2018.csv",
    # --- CMS ownership (Dec 2022 release, aligned with Compendium 2022) ---
    "cms_hospital_all_owners_2022-12.csv": "https://data.cms.gov/sites/default/files/2026-06/0692ea5b-6fc1-4aed-8b25-48f5e67b9951/Hospital_All_Owners_2022.12.30.csv",
    "cms_hospital_enrollments_2022-12.csv": "https://data.cms.gov/sites/default/files/2023-01/438b668c-0361-4482-9ac1-98fca829b310/Hospital_Enrollments_2022.12.30.csv",
    # --- CMS Provider of Services, Q4 2022 (hospital + non-hospital) ---
    "cms_pos_other_2022-12.csv": "https://data.cms.gov/sites/default/files/2023-01/77f97345-7163-4c5d-befd-f01935192b6d/POS_OTHER_DEC22.csv",
    # --- HCRIS-derived hospital cost report, FY2022 ---
    "cms_cost_report_2022.csv": "https://data.cms.gov/sites/default/files/2026-10/1be4ec08-6d9d-4e05-b871-c4d74fe81089/CostReport_2022_Final.csv",
    # --- Care Compare (current release; archived snapshots are not scriptable) ---
    "cc_hospital_general_info.csv": "provider-data:xubh-q36u",
    "cc_unplanned_visits.csv": "provider-data:632h-zaca",
    "cc_mspb.csv": "provider-data:rrqw-56er",
    # --- NPPES full monthly file (~1 GB zip) ---
    "nppes_full.zip": "https://download.cms.gov/nppes/NPPES_Data_Dissemination_November_2026_V2.zip",
}


def resolve(url: str) -> str:
    """provider-data:<id> -> current CSV download URL from the metastore API."""
    if not url.startswith("provider-data:"):
        return url
    ds_id = url.split(":", 1)[1]
    # curl rather than requests: the metastore returned an empty/non-JSON body
    # to python-requests but answers curl normally (issue I-08).
    res = subprocess.run(
        ["curl", "-sS", "-L",
         f"https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{ds_id}"],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(res.stdout)
    return meta["distribution"][0]["downloadURL"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str) -> dict:
    out = RAW / name
    real = resolve(url)
    if out.exists() and out.stat().st_size > 0:
        print(f"skip  {name}")
    else:
        print(f"get   {name} <- {real}", flush=True)
        tmp = out.with_suffix(out.suffix + ".part")
        for attempt in range(3):
            try:
                # Plain curl with its own default User-Agent (no browser spoofing).
                # If a site answers with a bot challenge we stop and ask for a
                # manual download instead of trying to get around it (issue I-07).
                res = subprocess.run(
                    ["curl", "-sS", "-L", "--retry", "2",
                     "-o", str(tmp), "-w", "%{http_code} %{content_type}", real],
                    capture_output=True, text=True, check=True,
                )
                code, _, ctype = res.stdout.partition(" ")
                if code != "200" or "html" in ctype:
                    raise RuntimeError(f"got {code} {ctype} instead of data")
                tmp.replace(out)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            print(f"FAILED {name}: download it manually from {real} and save it as "
                  f"{out}", file=sys.stderr)
            return {"file": name, "source": url, "url": real, "status": "manual-needed"}
    return {
        "file": name, "source": url, "url": real, "bytes": out.stat().st_size,
        "sha256": sha256(out), "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    only = set(sys.argv[1:])
    manifest_path = RAW / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in SOURCES.items():
        if only and name not in only:
            continue
        manifest[name] = fetch(name, url)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print("done:", len(manifest), "files")


if __name__ == "__main__":
    main()
