"""Stage 01: clean and normalize every source into 04-Data/clean/*.parquet.

Outputs
  hospitals.parquet     one row per CCN in the AHRQ 2018/2022 universe, with POS,
                        cost-report and Care Compare attributes
  systems.parquet       AHRQ systems (2018 and 2022 releases), one row per (year, id)
  labels.parquet        (ccn, year, health_sys_id) hospital->system ground truth
  owners.parquet        (ccn, owner_id, owner_type, role, name) from CMS All Owners
  enroll_npi.parquet    (ccn, npi) pairs from CMS Hospital Enrollments (ER labels)
  outcomes.parquet      star rating, hospital-wide readmission, MSPB, cost proxies
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN, ccn6, log, norm_addr, norm_name, phone10, read_cms_csv, save_metrics, zip5


def ahrq():
    labels, systems, hosp = [], [], []
    for year in (2018, 2022):
        lk = read_cms_csv(f"ahrq_hospital_linkage_{year}.csv")
        lk["ccn"] = ccn6(lk["ccn"])
        lk["year"] = year
        hosp.append(lk)
        lab = lk.loc[lk["health_sys_id"].notna(), ["ccn", "year", "health_sys_id"]]
        labels.append(lab)
        sy = read_cms_csv(f"ahrq_systems_{year}.csv")
        sy["year"] = year
        systems.append(sy[["health_sys_id", "health_sys_name", "health_sys_city", "health_sys_state",
                           "hosp_cnt", "acutehosp_cnt", "sys_beds", "sys_multistate", "year"]])
    labels = pd.concat(labels).drop_duplicates()
    systems = pd.concat(systems)
    systems["sys_name_norm"] = norm_name(systems["health_sys_name"])
    for c in ("hosp_cnt", "acutehosp_cnt", "sys_beds", "sys_multistate"):
        systems[c] = pd.to_numeric(systems[c], errors="coerce")
    hosp = pd.concat(hosp)
    # Keep the most recent record per CCN for attributes.
    hosp = hosp.sort_values("year").groupby("ccn", as_index=False).last()
    return labels, systems, hosp


def pos():
    p = read_cms_csv("cms_pos_other_2022-12.csv")
    p = p[p["PRVDR_CTGRY_CD"] == "01"].copy()  # 01 = hospital
    p["ccn"] = ccn6(p["PRVDR_NUM"])
    keep = {
        "ccn": "ccn", "FAC_NAME": "pos_name", "ST_ADR": "pos_street", "CITY_NAME": "pos_city",
        "STATE_CD": "pos_state", "ZIP_CD": "pos_zip", "PHNE_NUM": "pos_phone",
        "GNRL_CNTL_TYPE_CD": "pos_control", "CBSA_URBN_RRL_IND": "pos_urban",
        "BED_CNT": "pos_beds", "MLT_OWND_FAC_ORG_SW": "pos_multi_owned",
        "MLT_FAC_ORG_NAME": "pos_multi_org_name", "MDCL_SCHL_AFLTN_CD": "pos_med_school",
        "PRVDR_CTGRY_SBTYP_CD": "pos_subtype", "PGM_TRMNTN_CD": "pos_term", "CHOW_CNT": "pos_chow_cnt",
        "CHOW_DT": "pos_chow_dt",
    }
    p = p[list(keep)].rename(columns=keep)
    p = p.sort_values("pos_term").drop_duplicates("ccn")  # 00 = active sorts first
    p["pos_zip"] = zip5(p["pos_zip"])
    p["pos_phone"] = phone10(p["pos_phone"])
    p["pos_beds"] = pd.to_numeric(p["pos_beds"], errors="coerce")
    return p


def cost_report():
    c = read_cms_csv("cms_cost_report_2022.csv")
    c["ccn"] = ccn6(c["Provider CCN"])
    num = {
        "Number of Beds": "cr_beds", "Total Costs": "cr_total_costs",
        "Combined Outpatient + Inpatient Total Charges": "cr_total_charges",
        "Cost To Charge Ratio": "cr_cost_to_charge", "Net Patient Revenue": "cr_net_patient_rev",
        "Total Discharges (V + XVIII + XIX + Unknown)": "cr_discharges",
        "Number of Interns and Residents (FTE)": "cr_residents",
        "Total Patient Revenue": "cr_total_patient_rev",
    }
    out = c[["ccn", "Rural Versus Urban", "Type of Control"] + list(num)].rename(
        columns={**num, "Rural Versus Urban": "cr_rural", "Type of Control": "cr_control"})
    for v in num.values():
        out[v] = pd.to_numeric(out[v], errors="coerce")
    # Several reports per CCN are possible (partial years): keep the largest.
    out = out.sort_values("cr_total_costs").drop_duplicates("ccn", keep="last")
    # Price proxy: net patient revenue per discharge; markup: charges / costs.
    out["cr_markup"] = out["cr_total_charges"] / out["cr_total_costs"]
    out["cr_rev_per_dsch"] = out["cr_net_patient_rev"] / out["cr_discharges"]
    return out


def care_compare():
    g = read_cms_csv("cc_hospital_general_info.csv")
    g["ccn"] = ccn6(g["Facility ID"])
    g["star"] = pd.to_numeric(g["Hospital overall rating"], errors="coerce")
    g = g.rename(columns={"Facility Name": "cc_name", "Address": "cc_street", "City/Town": "cc_city",
                          "State": "cc_state", "ZIP Code": "cc_zip", "Telephone Number": "cc_phone",
                          "Hospital Type": "cc_type", "Hospital Ownership": "cc_ownership",
                          "Emergency Services": "cc_emergency"})
    g["cc_zip"] = zip5(g["cc_zip"])
    g["cc_phone"] = phone10(g["cc_phone"])
    g = g[["ccn", "cc_name", "cc_street", "cc_city", "cc_state", "cc_zip", "cc_phone", "cc_type",
           "cc_ownership", "cc_emergency", "star"]]

    u = read_cms_csv("cc_unplanned_visits.csv")
    u["ccn"] = ccn6(u["Facility ID"])
    hwr = u[u["Measure ID"] == "Hybrid_HWR"][["ccn", "Score"]].rename(columns={"Score": "readmit_hwr"})
    hwr["readmit_hwr"] = pd.to_numeric(hwr["readmit_hwr"], errors="coerce")

    m = read_cms_csv("cc_mspb.csv")
    m["ccn"] = ccn6(m["Facility ID"])
    m = m[m["Measure ID"] == "MSPB-1"][["ccn", "Score"]].rename(columns={"Score": "mspb"})
    m["mspb"] = pd.to_numeric(m["mspb"], errors="coerce")
    return g.merge(hwr, on="ccn", how="left").merge(m, on="ccn", how="left")


def owners_and_enrollments():
    e = read_cms_csv("cms_hospital_enrollments_2022-12.csv")
    e["ccn"] = ccn6(e["CCN"])
    enroll_npi = e[["ccn", "NPI"]].rename(columns={"NPI": "npi"}).dropna().drop_duplicates()
    enr = e[["ENROLLMENT ID", "ccn"]].drop_duplicates()

    o = read_cms_csv("cms_hospital_all_owners_2022-12.csv")
    o = o.merge(enr, on="ENROLLMENT ID", how="inner")
    o["owner_name"] = np.where(o["TYPE - OWNER"] == "O", o["ORGANIZATION NAME - OWNER"],
                               (o["FIRST NAME - OWNER"].fillna("") + " " + o["LAST NAME - OWNER"].fillna("")))
    o = o.rename(columns={"ASSOCIATE ID - OWNER": "owner_id", "TYPE - OWNER": "owner_type",
                          "ROLE CODE - OWNER": "role_code", "ROLE TEXT - OWNER": "role_text",
                          "STATE - OWNER": "owner_state", "PERCENTAGE OWNERSHIP": "pct"})
    o = o[["ccn", "owner_id", "owner_type", "role_code", "role_text", "owner_name", "owner_state", "pct"]]
    o["owner_name_norm"] = norm_name(o["owner_name"])
    return o.drop_duplicates(["ccn", "owner_id", "role_code"]), enroll_npi


def main():
    labels, systems, ahosp = ahrq()
    p, cr, cc = pos(), cost_report(), care_compare()
    owners, enroll_npi = owners_and_enrollments()

    h = ahosp[["ccn", "hospital_name", "hospital_street", "hospital_city", "hospital_state", "hospital_zip",
               "acutehosp_flag", "hos_beds", "hos_majteach", "hos_vmajteach", "hos_teachint", "hos_ownership",
               "hos_children", "hos_highuc", "year"]].rename(columns={"year": "last_ahrq_year"})
    h = h.merge(p, on="ccn", how="left").merge(cr, on="ccn", how="left").merge(cc, on="ccn", how="left")
    h["name_norm"] = norm_name(h["hospital_name"])
    h["street_norm"] = norm_addr(h["hospital_street"])
    h["zip"] = zip5(h["hospital_zip"]).fillna(h["pos_zip"])
    h["state"] = h["hospital_state"]
    h["city"] = h["hospital_city"].str.upper().str.strip()
    h["phone"] = h["pos_phone"].fillna(h["cc_phone"])
    for c in ("hos_beds", "hos_majteach", "hos_vmajteach", "hos_teachint", "acutehosp_flag", "hos_children",
              "hos_highuc", "hos_ownership"):
        h[c] = pd.to_numeric(h[c], errors="coerce")

    h.to_parquet(CLEAN / "hospitals.parquet", index=False)
    systems.to_parquet(CLEAN / "systems.parquet", index=False)
    labels.to_parquet(CLEAN / "labels.parquet", index=False)
    owners.to_parquet(CLEAN / "owners.parquet", index=False)
    enroll_npi.to_parquet(CLEAN / "enroll_npi.parquet", index=False)

    stats = {
        "hospitals": len(h),
        "hospitals_by_last_ahrq_year": h["last_ahrq_year"].value_counts().to_dict(),
        "labels_by_year": labels.groupby("year").size().to_dict(),
        "systems_by_year": systems.groupby("year").size().to_dict(),
        "pos_match_rate": float(h["pos_name"].notna().mean()),
        "cost_report_match_rate": float(h["cr_total_costs"].notna().mean()),
        "care_compare_match_rate": float(h["cc_name"].notna().mean()),
        "star_available": int(h["star"].notna().sum()),
        "readmit_available": int(h["readmit_hwr"].notna().sum()),
        "mspb_available": int(h["mspb"].notna().sum()),
        "owner_rows": len(owners),
        "owner_rows_by_type": owners["owner_type"].value_counts().to_dict(),
        "owner_hospitals": int(owners["ccn"].nunique()),
        "owner_hospitals_in_universe": int(owners["ccn"].isin(h["ccn"]).groupby(owners["ccn"]).first().sum()),
        "enroll_npi_pairs": len(enroll_npi),
        "pos_multi_owned_share": float((h["pos_multi_owned"] == "Y").mean()),
    }
    save_metrics("clean", stats)
    for k, v in stats.items():
        log(f"{k}: {v}")


if __name__ == "__main__":
    main()
