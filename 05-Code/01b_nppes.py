"""Stage 01b: filter the 11.8 GB NPPES file down to hospital organizations.

Never loads NPPES into pandas. The main CSV is extracted once, filtered with
DuckDB (entity type 2 + any hospital taxonomy 27x/28x), and then deleted; the
zip stays in raw/ as the source of record.

Outputs
  nppes_orgs.parquet     npi, names, practice address, phones, authorized official,
                         subpart/parent LBN, primary taxonomy
  nppes_otherid.parquet  (npi, other_id, type_code, state, issuer) unpivoted,
                         used for free NPI<->CCN labels
"""
from __future__ import annotations

import shutil
import zipfile

import duckdb

from common import CLEAN, RAW, log, save_metrics

ZIP = RAW / "nppes_full.zip"
WORK = RAW / "nppes_extract"


def main():
    WORK.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        member = next(n for n in z.namelist() if n.startswith("npidata_pfile") and "fileheader" not in n)
        csv_path = WORK / member
        if not csv_path.exists():
            log(f"extracting {member} ...")
            with z.open(member) as src, csv_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 24)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    tax = " OR ".join(
        f"(substr(\"Healthcare Provider Taxonomy Code_{i}\",1,2) IN ('27','28'))" for i in range(1, 16))
    prim = " ".join(
        f"WHEN \"Healthcare Provider Primary Taxonomy Switch_{i}\"='Y' THEN \"Healthcare Provider Taxonomy Code_{i}\""
        for i in range(1, 16))
    src = f"read_csv('{csv_path.as_posix()}', all_varchar=true, header=true, quote='\"', parallel=true)"
    log("filtering organizations with hospital taxonomies ...")
    con.execute(f"""
        CREATE TABLE orgs AS SELECT * FROM {src}
        WHERE "Entity Type Code"='2' AND ({tax})
    """)
    n = con.execute("SELECT count(*) FROM orgs").fetchone()[0]
    log(f"  {n} hospital-taxonomy organizations")

    con.execute(f"""
        COPY (SELECT
            "NPI" AS npi,
            "Provider Organization Name (Legal Business Name)" AS lbn,
            "Provider Other Organization Name" AS other_name,
            "Provider First Line Business Practice Location Address" AS pl_street,
            "Provider Business Practice Location Address City Name" AS pl_city,
            "Provider Business Practice Location Address State Name" AS pl_state,
            substr("Provider Business Practice Location Address Postal Code",1,5) AS pl_zip,
            "Provider Business Practice Location Address Telephone Number" AS pl_phone,
            "Authorized Official Last Name" AS ao_last,
            "Authorized Official First Name" AS ao_first,
            "Authorized Official Title or Position" AS ao_title,
            "Authorized Official Telephone Number" AS ao_phone,
            "Is Organization Subpart" AS is_subpart,
            "Parent Organization LBN" AS parent_lbn,
            "Employer Identification Number (EIN)" AS ein,
            "NPI Deactivation Date" AS deactivated,
            CASE {prim} ELSE "Healthcare Provider Taxonomy Code_1" END AS primary_taxonomy
        FROM orgs) TO '{(CLEAN / "nppes_orgs.parquet").as_posix()}' (FORMAT parquet)
    """)

    unions = " UNION ALL ".join(
        f"""SELECT "NPI" AS npi, "Other Provider Identifier_{i}" AS other_id,
                   "Other Provider Identifier Type Code_{i}" AS type_code,
                   "Other Provider Identifier State_{i}" AS state,
                   "Other Provider Identifier Issuer_{i}" AS issuer
            FROM orgs WHERE "Other Provider Identifier_{i}" IS NOT NULL""" for i in range(1, 51))
    con.execute(f"COPY ({unions}) TO '{(CLEAN / 'nppes_otherid.parquet').as_posix()}' (FORMAT parquet)")
    m = con.execute(f"SELECT count(*), count(DISTINCT npi) FROM '{(CLEAN / 'nppes_otherid.parquet').as_posix()}'").fetchone()
    tax_counts = con.execute(f"""SELECT substr(primary_taxonomy,1,4) t, count(*) c
        FROM '{(CLEAN / 'nppes_orgs.parquet').as_posix()}' GROUP BY 1 ORDER BY 2 DESC LIMIT 12""").fetchall()
    save_metrics("nppes", {"hospital_orgs": n, "other_id_rows": m[0], "other_id_npis": m[1],
                           "primary_taxonomy_top": {t: c for t, c in tax_counts},
                           "source_file": member})
    log(f"  other-id rows {m[0]} across {m[1]} NPIs")
    con.close()
    csv_path.unlink()  # 11.8 GB; the zip remains the source of record
    log("deleted extracted CSV")


if __name__ == "__main__":
    main()
