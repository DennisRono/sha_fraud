import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
#  CONFIGURATION & THRESHOLDS
CONFIG = {
    # Biological constraints
    "min_inter_delivery_days": 270,          # Minimum days between deliveries (9 months)
    "max_lifetime_deliveries": 15,           # Absolute maximum deliveries per female
    "max_delivery_age": 55,                  # Age above which delivery is biologically implausible
    "min_delivery_age": 10,                  # Age below which delivery is implausible
    "max_major_surgeries_per_month": 3,      # Max major surgeries one patient can undergo/month
    "max_minor_surgeries_per_month": 6,      # Max minor surgeries one patient can undergo/month
    "max_surgeries_per_day_per_patient": 2,  # Hard daily cap per patient
    "min_inter_surgery_days_major": 21,      # Minimum recovery days between major surgeries
    # Facility operational constraints
    "max_theater_hours_per_day": 8,          # Hours a theater can realistically operate
    "avg_major_surgery_duration_min": 120,   # Minutes for major surgery
    "avg_minor_surgery_duration_min": 45,    # Minutes for minor surgery
    "max_facility_occupancy_rate": 0.95,     # Absolute ceiling occupancy rate
    "suspicious_occupancy_rate": 0.85,       # Above this triggers a soft flag
    "max_outpatients_per_doctor_per_day": 30,# Consultant throughput ceiling
    "max_inpatients_per_doctor_per_day": 10, # Ward round throughput ceiling
    # Statistical thresholds
    "zscore_monitor_threshold": 2.0,
    "zscore_investigate_threshold": 3.0,
    "zscore_mandatory_audit_threshold": 4.0,
    "benford_pvalue_threshold": 0.05,        # Chi-square p-value for Benford's law
    "month_end_stuffing_ratio": 0.40,        # >40% claims in last 5 days = suspicious
    # Fraud Risk Score (FRS) thresholds
    "frs_immediate_audit": 15,
    "frs_enhanced_monitoring": 8,
}
# Fraud flag weights
FLAG_WEIGHTS = {
    "BIOLOGICAL_IMPOSSIBLE":        10,
    "CAPACITY_OVERFLOW":             8,
    "ZSCORE_EXTREME":                7,
    "NETWORK_COLLUSION":             7,
    "DUPLICATE_CLAIM":               6,
    "BENFORD_DEVIATION":             5,
    "TEMPORAL_CLUSTERING":           4,
    "STAFFING_MISMATCH":             4,
    "CLINICAL_CODE_ANOMALY":         5,
    "IMPOSSIBLE_COMBINATION":        9,
    "GHOST_PATIENT":                 6,
    "UPCODING":                      4,
    "STATISTICAL_OUTLIER":           3,
}

#  HELPER UTILITIES

def compute_age(dob: pd.Series, reference_date: pd.Series) -> pd.Series:
    """Compute age in years from date-of-birth Series."""
    return ((reference_date - dob).dt.days / 365.25).round(1)

def benford_expected_freq() -> np.ndarray:
    """Return Benford's Law expected frequency for digits 1–9."""
    return np.array([np.log10(1 + 1/d) for d in range(1, 10)])

def leading_digit(series: pd.Series) -> pd.Series:
    """Extract leading digit from a numeric series."""
    return series.abs().astype(str).str.lstrip("0").str[0].astype(int)

def flag_record(flags: list, claim_id, flag_type: str, detail: str, severity: str = "HIGH"):
    """Append a structured flag record."""
    flags.append({
        "claim_id": claim_id,
        "flag_type": flag_type,
        "severity": severity,
        "weight": FLAG_WEIGHTS.get(flag_type, 1),
        "detail": detail,
    })

#  MODULE 1 — BIOLOGICAL IMPOSSIBILITY CHECKS

def check_biological_impossibilities(claims: pd.DataFrame, members: pd.DataFrame) -> list:
    """
    Detect claims that violate absolute biological and physiological constraints.
    Required columns:
      claims  : claim_id, member_id, service_code, claim_date, procedure_type
      members : member_id, dob, sex
    """
    flags = []
    df = claims.merge(members[["member_id", "dob", "sex"]], on="member_id", how="left")
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["age_at_claim"] = compute_age(df["dob"], df["claim_date"])
    # ── 1a. Delivery checks ──────────────────────────────────────────────
    deliveries = df[df["procedure_type"].str.upper().isin(["DELIVERY", "C_SECTION", "CAESAREAN"])].copy()
    deliveries_sorted = deliveries.sort_values(["member_id", "claim_date"])
    for member_id, grp in deliveries_sorted.groupby("member_id"):
        grp = grp.reset_index(drop=True)
        member_sex = grp["sex"].iloc[0]
        # Male claiming delivery
        if str(member_sex).upper() in ["M", "MALE"]:
            for _, row in grp.iterrows():
                flag_record(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id} is MALE but claimed a delivery/obstetric procedure.")
        # Multiple deliveries same day
        same_day = grp.groupby("claim_date").size()
        for date, count in same_day.items():
            if count > 1:
                ids = grp[grp["claim_date"] == date]["claim_id"].tolist()
                for cid in ids:
                    flag_record(flags, cid, "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id} has {count} delivery claims on {date.date()} — biologically impossible.")
        # Deliveries within 270-day window
        dates = grp["claim_date"].tolist()
        claim_ids = grp["claim_id"].tolist()
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i-1]).days
            if gap_days < CONFIG["min_inter_delivery_days"]:
                flag_record(flags, claim_ids[i], "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: {gap_days} days between deliveries (minimum = {CONFIG['min_inter_delivery_days']} days).")
        # Lifetime delivery count
        if len(grp) > CONFIG["max_lifetime_deliveries"]:
            for _, row in grp.iterrows():
                flag_record(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: {len(grp)} lifetime deliveries claimed (max plausible = {CONFIG['max_lifetime_deliveries']}).")
        # Age at delivery
        for _, row in grp.iterrows():
            age = row["age_at_claim"]
            if pd.notna(age):
                if age > CONFIG["max_delivery_age"]:
                    flag_record(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id} aged {age:.1f} years claimed delivery (post-menopausal threshold = {CONFIG['max_delivery_age']}).")
                if age < CONFIG["min_delivery_age"]:
                    flag_record(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id} aged {age:.1f} years claimed delivery (below minimum age {CONFIG['min_delivery_age']}).")
    # ── 1b. Surgical volume per patient ──────────────────────────────────
    surgeries = df[df["procedure_type"].str.upper().isin(["MAJOR_SURGERY", "MINOR_SURGERY"])].copy()
    surgeries["year_month"] = surgeries["claim_date"].dt.to_period("M")
    for (member_id, ym, ptype), grp in surgeries.groupby(["member_id", "year_month", "procedure_type"]):
        count = len(grp)
        limit = (CONFIG["max_major_surgeries_per_month"] if ptype.upper() == "MAJOR_SURGERY"
                 else CONFIG["max_minor_surgeries_per_month"])
        if count > limit:
            for _, row in grp.iterrows():
                flag_record(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: {count} {ptype} claims in {ym} (limit = {limit}/month).")
    # Same-day surgeries per patient
    for (member_id, date), grp in surgeries.groupby(["member_id", "claim_date"]):
        if len(grp) > CONFIG["max_surgeries_per_day_per_patient"]:
            for _, row in grp.iterrows():
                flag_record(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: {len(grp)} surgeries claimed on {date.date()} (daily cap = {CONFIG['max_surgeries_per_day_per_patient']}).")
    # Inter-surgery recovery gap (major surgeries only)
    major = surgeries[surgeries["procedure_type"].str.upper() == "MAJOR_SURGERY"].sort_values(["member_id", "claim_date"])
    for member_id, grp in major.groupby("member_id"):
        grp = grp.reset_index(drop=True)
        dates = grp["claim_date"].tolist()
        cids = grp["claim_id"].tolist()
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap < CONFIG["min_inter_surgery_days_major"]:
                flag_record(flags, cids[i], "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: Only {gap} days between major surgeries (minimum recovery = {CONFIG['min_inter_surgery_days_major']} days).")
    return flags

#  MODULE 2 — FACILITY CAPACITY CHECKS

def check_facility_capacity(claims: pd.DataFrame, facilities: pd.DataFrame) -> list:
    """
    Detect facilities claiming more activity than their physical capacity allows.
    Required columns:
      claims     : claim_id, facility_id, claim_date, procedure_type,
                   inpatient_days, claim_amount
      facilities : facility_id, bed_count, theater_count, registered_doctors
    """
    flags = []
    df = claims.merge(facilities, on="facility_id", how="left")
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["year_month"] = df["claim_date"].dt.to_period("M")
    for (facility_id, ym), grp in df.groupby(["facility_id", "year_month"]):
        bed_count = grp["bed_count"].iloc[0]
        theater_count = grp["theater_count"].iloc[0]
        n_doctors = grp["registered_doctors"].iloc[0]
        days_in_month = ym.days_in_month
        # ── 2a. Inpatient-day overflow ────────────────────────────────
        total_inpatient_days = grp["inpatient_days"].sum()
        max_possible = bed_count * days_in_month
        suspicious_limit = max_possible * CONFIG["suspicious_occupancy_rate"]
        if total_inpatient_days > max_possible:
            ratio = total_inpatient_days / max_possible
            claim_ids = grp["claim_id"].tolist()
            for cid in claim_ids:
                flag_record(flags, cid, "CAPACITY_OVERFLOW",
                    f"Facility {facility_id} [{ym}]: {total_inpatient_days} inpatient-days claimed "
                    f"vs max possible {max_possible} ({ratio:.2f}× capacity — PHYSICALLY IMPOSSIBLE).")
        elif total_inpatient_days > suspicious_limit:
            claim_ids = grp["claim_id"].tolist()
            for cid in claim_ids:
                flag_record(flags, cid, "CAPACITY_OVERFLOW",
                    f"Facility {facility_id} [{ym}]: {total_inpatient_days} inpatient-days "
                    f"({total_inpatient_days/max_possible*100:.1f}% occupancy — above {CONFIG['suspicious_occupancy_rate']*100:.0f}% threshold).",
                    severity="MEDIUM")
        # ── 2b. Theater throughput ────────────────────────────────────
        surgeries = grp[grp["procedure_type"].str.upper().isin(["MAJOR_SURGERY", "MINOR_SURGERY"])]
        for date, day_grp in surgeries.groupby(df["claim_date"]):
            n_major = (day_grp["procedure_type"].str.upper() == "MAJOR_SURGERY").sum()
            n_minor = (day_grp["procedure_type"].str.upper() == "MINOR_SURGERY").sum()
            available_min = CONFIG["max_theater_hours_per_day"] * 60 * theater_count
            used_min = (n_major * CONFIG["avg_major_surgery_duration_min"] +
                        n_minor * CONFIG["avg_minor_surgery_duration_min"])
            if used_min > available_min:
                for cid in day_grp["claim_id"].tolist():
                    flag_record(flags, cid, "CAPACITY_OVERFLOW",
                        f"Facility {facility_id} on {pd.Timestamp(date).date()}: "
                        f"{n_major} major + {n_minor} minor surgeries requires {used_min} mins "
                        f"but only {available_min} mins available ({theater_count} theater(s)).")
        # ── 2c. Doctor-to-patient throughput ─────────────────────────
        outpatients_month = (grp["procedure_type"].str.upper() == "OUTPATIENT").sum()
        max_outpatients_month = n_doctors * CONFIG["max_outpatients_per_doctor_per_day"] * days_in_month
        if outpatients_month > max_outpatients_month:
            for cid in grp[grp["procedure_type"].str.upper() == "OUTPATIENT"]["claim_id"].tolist():
                flag_record(flags, cid, "STAFFING_MISMATCH",
                    f"Facility {facility_id} [{ym}]: {outpatients_month} outpatients claimed "
                    f"but {n_doctors} doctor(s) can handle max {max_outpatients_month}.")
    return flags

#  MODULE 3 — STATISTICAL DISTRIBUTION CHECKS

def check_statistical_anomalies(claims: pd.DataFrame, facilities: pd.DataFrame) -> list:
    """
    Apply Benford's Law, Z-score peer analysis, and temporal pattern analysis.
    """
    flags = []
    df = claims.merge(facilities[["facility_id", "facility_level", "county", "ownership"]], on="facility_id", how="left")
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["year_month"] = df["claim_date"].dt.to_period("M")
    # ── 3a. Benford's Law per facility ───────────────────────────────
    benford_expected = benford_expected_freq()
    for facility_id, grp in df.groupby("facility_id"):
        if len(grp) < 100:
            continue  # Insufficient data for reliable Benford analysis
        amounts = grp["claim_amount"].dropna()
        amounts = amounts[amounts > 0]
        first_digits = leading_digit(amounts)
        observed_counts = first_digits.value_counts().reindex(range(1, 10), fill_value=0).values
        total = observed_counts.sum()
        if total < 50:
            continue
        expected_counts = benford_expected * total
        chi2_stat, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)
        if p_value < CONFIG["benford_pvalue_threshold"]:
            for cid in grp["claim_id"].tolist():
                flag_record(flags, cid, "BENFORD_DEVIATION",
                    f"Facility {facility_id}: Claim amounts deviate from Benford's Law "
                    f"(χ²={chi2_stat:.2f}, p={p_value:.4f}) — possible fabrication.",
                    severity="MEDIUM")
    # ── 3b. Z-score peer comparison ──────────────────────────────────
    monthly_volume = df.groupby(["facility_id", "year_month", "facility_level", "county"]).agg(
        claim_count=("claim_id", "count"),
        total_amount=("claim_amount", "sum")
    ).reset_index()
    for (level, county, ym), peer_grp in monthly_volume.groupby(["facility_level", "county", "year_month"]):
        if len(peer_grp) < 3:
            continue
        mean_vol = peer_grp["claim_count"].mean()
        std_vol = peer_grp["claim_count"].std()
        if std_vol == 0:
            continue
        peer_grp = peer_grp.copy()
        peer_grp["z_score"] = (peer_grp["claim_count"] - mean_vol) / std_vol
        for _, row in peer_grp[peer_grp["z_score"].abs() > CONFIG["zscore_investigate_threshold"]].iterrows():
            severity = ("HIGH" if abs(row["z_score"]) >= CONFIG["zscore_mandatory_audit_threshold"]
                        else "MEDIUM")
            flag_type = "ZSCORE_EXTREME" if abs(row["z_score"]) >= CONFIG["zscore_mandatory_audit_threshold"] else "STATISTICAL_OUTLIER"
            facility_claims = df[(df["facility_id"] == row["facility_id"]) &
                                  (df["year_month"] == row["year_month"])]["claim_id"].tolist()
            for cid in facility_claims:
                flag_record(flags, cid, flag_type,
                    f"Facility {row['facility_id']} [{ym}]: Z-score = {row['z_score']:.2f} "
                    f"(claims: {row['claim_count']} vs peer mean: {mean_vol:.1f}) in Level-{level}, {county} county.",
                    severity=severity)
    # ── 3c. Month-end claim stuffing ─────────────────────────────────
    df["day_of_month"] = df["claim_date"].dt.day
    df["days_in_month"] = df["claim_date"].dt.days_in_month
    df["is_last_5_days"] = df["day_of_month"] > (df["days_in_month"] - 5)
    for (facility_id, ym), grp in df.groupby(["facility_id", "year_month"]):
        total = len(grp)
        last5_count = grp["is_last_5_days"].sum()
        ratio = last5_count / total if total > 0 else 0
        if ratio > CONFIG["month_end_stuffing_ratio"] and total > 20:
            for cid in grp[grp["is_last_5_days"]]["claim_id"].tolist():
                flag_record(flags, cid, "TEMPORAL_CLUSTERING",
                    f"Facility {facility_id} [{ym}]: {last5_count}/{total} claims ({ratio*100:.1f}%) "
                    f"in last 5 days of month — month-end stuffing pattern (threshold = {CONFIG['month_end_stuffing_ratio']*100:.0f}%).",
                    severity="MEDIUM")
    # ── 3d. Sudden volume spikes ─────────────────────────────────────
    monthly_counts = df.groupby(["facility_id", "year_month"]).size().reset_index(name="count")
    monthly_counts["year_month_dt"] = monthly_counts["year_month"].dt.to_timestamp()
    monthly_counts = monthly_counts.sort_values(["facility_id", "year_month_dt"])
    for facility_id, grp in monthly_counts.groupby("facility_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < 4:
            continue
        rolling_mean = grp["count"].rolling(window=3, min_periods=2).mean().shift(1)
        rolling_std = grp["count"].rolling(window=3, min_periods=2).std().shift(1)
        grp["z_spike"] = (grp["count"] - rolling_mean) / rolling_std.replace(0, np.nan)
        for _, row in grp[grp["z_spike"] > CONFIG["zscore_investigate_threshold"]].iterrows():
            month_claims = df[(df["facility_id"] == facility_id) &
                               (df["year_month"] == row["year_month"])]["claim_id"].tolist()
            for cid in month_claims:
                flag_record(flags, cid, "STATISTICAL_OUTLIER",
                    f"Facility {facility_id}: Sudden volume spike in {row['year_month']} "
                    f"({row['count']} claims, Z-spike = {row['z_spike']:.2f} vs 3-month rolling avg).",
                    severity="MEDIUM")
    return flags

#  MODULE 4 — DUPLICATE CLAIM DETECTION

def check_duplicate_claims(claims: pd.DataFrame) -> list:
    """
    Detect exact and near-duplicate claim submissions.
    """
    flags = []
    df = claims.copy()
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    # ── 4a. Exact duplicates ──────────────────────────────────────────
    dupe_keys = ["member_id", "facility_id", "service_code", "claim_date", "claim_amount"]
    exact_dupes = df[df.duplicated(subset=dupe_keys, keep=False)]
    for cid in exact_dupes["claim_id"].tolist():
        flag_record(flags, cid, "DUPLICATE_CLAIM",
            f"Claim {cid}: Exact duplicate detected (same member, facility, service, date, and amount).")
    # ── 4b. Near-duplicates (same member+service within 3 days, different amounts) ──
    df_sorted = df.sort_values(["member_id", "service_code", "claim_date"])
    for (member_id, service_code), grp in df_sorted.groupby(["member_id", "service_code"]):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            for j in range(i+1, len(grp)):
                day_diff = abs((grp.loc[j, "claim_date"] - grp.loc[i, "claim_date"]).days)
                if day_diff <= 3 and day_diff > 0:
                    flag_record(flags, grp.loc[j, "claim_id"], "DUPLICATE_CLAIM",
                        f"Near-duplicate: Member {member_id}, service {service_code} claimed "
                        f"twice within {day_diff} day(s) — possible resubmission with modified amount.",
                        severity="MEDIUM")
    return flags

#  MODULE 5 — CLINICAL CODING ANOMALIES

# Mutually exclusive same-day procedure pairs
IMPOSSIBLE_COMBINATIONS = [
    ("APPENDECTOMY", "COLONOSCOPY",        "Cannot perform colonoscopy prep and appendectomy same day"),
    ("GENERAL_ANESTHESIA", "SAME_DAY_DISCHARGE", "Major GA procedure with same-day discharge is clinically unsafe"),
    ("MALE", "HYSTERECTOMY",              "Hysterectomy claimed for male member"),
    ("MALE", "CERVICAL_CANCER_SCREEN",    "Cervical cancer screening claimed for male member"),
    ("MALE", "DELIVERY",                  "Delivery claimed for male member"),
    ("MALE", "ANTENATAL",                 "Antenatal care claimed for male member"),
]
LEVEL2_RESTRICTED_PROCEDURES = [
    "ORGAN_TRANSPLANT", "NEUROSURGERY", "CARDIAC_BYPASS", "ICU_VENTILATION"
]
def check_clinical_coding_anomalies(claims: pd.DataFrame, members: pd.DataFrame, facilities: pd.DataFrame) -> list:
    """
    Detect impossible procedure combinations, wrong-sex procedures,
    upcoding, and procedures beyond facility capability.
    """
    flags = []
    df = claims.merge(members[["member_id", "sex"]], on="member_id", how="left")
    df = df.merge(facilities[["facility_id", "facility_level"]], on="facility_id", how="left")
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["procedure_upper"] = df["procedure_type"].str.upper()
    df["service_upper"] = df["service_code"].str.upper() if "service_code" in df.columns else df["procedure_upper"]
    # ── 5a. Sex-procedure mismatches ─────────────────────────────────
    male_members = df[df["sex"].str.upper().isin(["M", "MALE"])]
    female_only_procs = ["HYSTERECTOMY", "DELIVERY", "C_SECTION", "CAESAREAN",
                         "ANTENATAL", "CERVICAL_CANCER_SCREEN", "OVARIAN"]
    for proc in female_only_procs:
        mismatched = male_members[male_members["procedure_upper"].str.contains(proc, na=False)]
        for _, row in mismatched.iterrows():
            flag_record(flags, row["claim_id"], "IMPOSSIBLE_COMBINATION",
                f"Male member {row['member_id']} claimed female-only procedure: {proc}.")
    # ── 5b. Impossible same-day combinations ─────────────────────────
    for (member_id, date), day_grp in df.groupby(["member_id", df["claim_date"].dt.date]):
        procedures_today = day_grp["procedure_upper"].tolist()
        for (proc_a, proc_b, reason) in IMPOSSIBLE_COMBINATIONS:
            a_present = any(proc_a in p for p in procedures_today)
            b_present = any(proc_b in p for p in procedures_today)
            if a_present and b_present:
                for cid in day_grp["claim_id"].tolist():
                    flag_record(flags, cid, "IMPOSSIBLE_COMBINATION",
                        f"Member {member_id} on {date}: {reason}.")
    # ── 5c. Procedures beyond facility level ─────────────────────────
    level2_facilities = df[df["facility_level"].isin([2, "2", "Level 2"])]
    for proc in LEVEL2_RESTRICTED_PROCEDURES:
        over_scoped = level2_facilities[level2_facilities["procedure_upper"].str.contains(proc, na=False)]
        for _, row in over_scoped.iterrows():
            flag_record(flags, row["claim_id"], "CLINICAL_CODE_ANOMALY",
                f"Facility {row['facility_id']} (Level 2) claimed {proc} — beyond Level 2 scope.")
    # ── 5d. Upcoding: excess high-complexity consultations ────────────
    consultations = df[df["procedure_upper"].str.contains("CONSULT|OUTPATIENT|OPD", na=False)]
    for facility_id, grp in consultations.groupby("facility_id"):
        if "complexity" in grp.columns:
            total = len(grp)
            high_complexity = (grp["complexity"] >= 4).sum()
            high_ratio = high_complexity / total if total > 0 else 0
            if high_ratio > 0.60 and total > 30:
                for cid in grp[grp["complexity"] >= 4]["claim_id"].tolist():
                    flag_record(flags, cid, "UPCODING",
                        f"Facility {facility_id}: {high_ratio*100:.1f}% of consultations coded as high-complexity "
                        f"(expected < 30% for primary care). Possible upcoding.",
                        severity="MEDIUM")
    return flags

#  MODULE 6 — GHOST PATIENT DETECTION

def check_ghost_patients(claims: pd.DataFrame, members: pd.DataFrame) -> list:
    """
    Identify members exhibiting ghost patient patterns:
      - Only high-cost claims, no preventive / routine care
      - Multiple members at identical address + phone
      - Bulk-registered members (same agent, same date)
    """
    flags = []
    df = claims.merge(members, on="member_id", how="left")
    # ── 6a. No preventive care — only high-cost procedures ────────────
    low_acuity = ["IMMUNIZATION", "WELLNESS_VISIT", "ROUTINE_CHECKUP", "ANTENATAL_VISIT",
                  "FAMILY_PLANNING", "OUTPATIENT"]
    high_cost = ["MAJOR_SURGERY", "ICU", "DIALYSIS", "CHEMOTHERAPY", "ORGAN_TRANSPLANT"]
    for member_id, grp in df.groupby("member_id"):
        proc_list = grp["procedure_type"].str.upper().tolist()
        has_low_acuity = any(any(la in p for la in low_acuity) for p in proc_list)
        has_high_cost = any(any(hc in p for hc in high_cost) for p in proc_list)
        if has_high_cost and not has_low_acuity and len(grp) >= 3:
            for cid in grp["claim_id"].tolist():
                flag_record(flags, cid, "GHOST_PATIENT",
                    f"Member {member_id}: Only high-cost procedures claimed, zero preventive/routine care — ghost patient pattern.",
                    severity="MEDIUM")
    # ── 6b. Multiple members — same address and phone ─────────────────
    if "address" in members.columns and "phone" in members.columns:
        address_phone_groups = members.groupby(["address", "phone"])["member_id"].apply(list)
        suspicious_addresses = address_phone_groups[address_phone_groups.apply(len) > 4]
        suspicious_member_ids = [mid for mids in suspicious_addresses for mid in mids]
        suspicious_claims = df[df["member_id"].isin(suspicious_member_ids)]
        for cid in suspicious_claims["claim_id"].tolist():
            flag_record(flags, cid, "GHOST_PATIENT",
                f"Member shares address+phone with 4+ other members — possible ghost beneficiary cluster.",
                severity="MEDIUM")
    # ── 6c. Bulk registration (same agent, same date) ─────────────────
    if "registration_agent_id" in members.columns and "registration_date" in members.columns:
        members["registration_date"] = pd.to_datetime(members["registration_date"])
        bulk = members.groupby(["registration_agent_id", "registration_date"]).size().reset_index(name="count")
        bulk_suspicious = bulk[bulk["count"] > 20]
        for _, row in bulk_suspicious.iterrows():
            suspicious_mids = members[
                (members["registration_agent_id"] == row["registration_agent_id"]) &
                (members["registration_date"] == row["registration_date"])
            ]["member_id"].tolist()
            suspicious_claims = df[df["member_id"].isin(suspicious_mids)]
            for cid in suspicious_claims["claim_id"].tolist():
                flag_record(flags, cid, "GHOST_PATIENT",
                    f"Agent {row['registration_agent_id']} bulk-registered {row['count']} members on {row['registration_date'].date()} — possible ghost enrollment.",
                    severity="HIGH")
    return flags

#  MODULE 7 — NETWORK COLLUSION DETECTION

def check_network_collusion(claims: pd.DataFrame) -> list:
    """
    Identify suspicious member-facility relationships that suggest collusion.
    """
    flags = []
    df = claims.copy()
    # ── 7a. Member over-concentration at single facility ─────────────
    member_facility = df.groupby(["member_id", "facility_id"]).size().reset_index(name="visit_count")
    member_totals = df.groupby("member_id").size().reset_index(name="total_claims")
    mf_merged = member_facility.merge(member_totals, on="member_id")
    mf_merged["concentration_ratio"] = mf_merged["visit_count"] / mf_merged["total_claims"]
    suspicious_concentration = mf_merged[
        (mf_merged["concentration_ratio"] > 0.95) &  # >95% claims at ONE facility
        (mf_merged["total_claims"] > 10)             # enough volume to be meaningful
    ]
    for _, row in suspicious_concentration.iterrows():
        member_claims = df[(df["member_id"] == row["member_id"]) &
                           (df["facility_id"] == row["facility_id"])]["claim_id"].tolist()
        for cid in member_claims:
            flag_record(flags, cid, "NETWORK_COLLUSION",
                f"Member {row['member_id']}: {row['concentration_ratio']*100:.1f}% of {row['total_claims']} claims "
                f"exclusively at Facility {row['facility_id']} across all conditions — unnatural loyalty.",
                severity="MEDIUM")
    # ── 7b. Agent linked to high-claim members at same facility ───────
    if "registration_agent_id" in df.columns:
        agent_facility = df.groupby(["registration_agent_id", "facility_id"]).agg(
            total_amount=("claim_amount", "sum"),
            unique_members=("member_id", "nunique")
        ).reset_index()
        agent_totals = df.groupby("registration_agent_id")["claim_amount"].sum().reset_index(name="agent_total")
        af = agent_facility.merge(agent_totals, on="registration_agent_id")
        af["facility_concentration"] = af["total_amount"] / af["agent_total"]
        for _, row in af[af["facility_concentration"] > 0.80].iterrows():
            agent_claims = df[df["registration_agent_id"] == row["registration_agent_id"]]["claim_id"].tolist()
            for cid in agent_claims:
                flag_record(flags, cid, "NETWORK_COLLUSION",
                    f"Agent {row['registration_agent_id']}: {row['facility_concentration']*100:.1f}% "
                    f"of enrolled members' claims concentrated at Facility {row['facility_id']} — possible collusion ring.",
                    severity="HIGH")
    return flags

#  FRAUD RISK SCORING ENGINE

def compute_fraud_risk_scores(all_flags: list, claims: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all flags per claim into a Fraud Risk Score (FRS).
    Classify each claim into risk tiers.
    """
    if not all_flags:
        claims_out = claims[["claim_id"]].copy()
        claims_out["fraud_risk_score"] = 0
        claims_out["risk_tier"] = "CLEAR"
        claims_out["flags"] = ""
        claims_out["flag_count"] = 0
        return claims_out
    flags_df = pd.DataFrame(all_flags)
    score_by_claim = flags_df.groupby("claim_id").agg(
        fraud_risk_score=("weight", "sum"),
        flag_count=("flag_type", "count"),
        flags=("flag_type", lambda x: " | ".join(sorted(set(x)))),
        details=("detail", lambda x: " || ".join(x))
    ).reset_index()
    result = claims[["claim_id"]].merge(score_by_claim, on="claim_id", how="left")
    result["fraud_risk_score"] = result["fraud_risk_score"].fillna(0)
    result["flag_count"] = result["flag_count"].fillna(0).astype(int)
    result["flags"] = result["flags"].fillna("")
    result["details"] = result["details"].fillna("")
    def classify(score):
        if score >= CONFIG["frs_immediate_audit"]:
            return "IMMEDIATE_AUDIT"
        elif score >= CONFIG["frs_enhanced_monitoring"]:
            return "ENHANCED_MONITORING"
        elif score > 0:
            return "WATCH"
        else:
            return "CLEAR"
    result["risk_tier"] = result["fraud_risk_score"].apply(classify)
    return result.sort_values("fraud_risk_score", ascending=False).reset_index(drop=True)

#  MAIN ORCHESTRATOR

def run_fraud_detection(
    claims: pd.DataFrame,
    members: pd.DataFrame,
    facilities: pd.DataFrame,
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full SHA fraud detection pipeline across all claims.
    Parameters
    ----------
    claims     : DataFrame of all claim records
    members    : DataFrame of beneficiary/member records
    facilities : DataFrame of registered facility details
    verbose    : Print progress and summary statistics
    Returns
    -------
    scored_claims : DataFrame — every claim with fraud risk score + tier
    flags_df      : DataFrame — detailed flag records for auditors
    """
    if verbose:
        print("=" * 65)
        print("  SHA KENYA — FRAUD DETECTION ENGINE")
        print("=" * 65)
        print(f"  Claims   : {len(claims):,}")
        print(f"  Members  : {len(members):,}")
        print(f"  Facilities: {len(facilities):,}")
        print("-" * 65)
    all_flags = []
    modules = [
        ("Module 1 — Biological Impossibilities",
         lambda: check_biological_impossibilities(claims, members)),
        ("Module 2 — Facility Capacity Violations",
         lambda: check_facility_capacity(claims, facilities)),
        ("Module 3 — Statistical Distribution Anomalies",
         lambda: check_statistical_anomalies(claims, facilities)),
        ("Module 4 — Duplicate Claims",
         lambda: check_duplicate_claims(claims)),
        ("Module 5 — Clinical Coding Anomalies",
         lambda: check_clinical_coding_anomalies(claims, members, facilities)),
        ("Module 6 — Ghost Patient Detection",
         lambda: check_ghost_patients(claims, members)),
        ("Module 7 — Network Collusion",
         lambda: check_network_collusion(claims)),
    ]
    for name, fn in modules:
        if verbose:
            print(f"  Running {name}...", end=" ", flush=True)
        module_flags = fn()
        all_flags.extend(module_flags)
        if verbose:
            print(f"{len(module_flags)} flags raised.")
    if verbose:
        print("-" * 65)
        print(f"  Total flags raised: {len(all_flags):,}")
    scored_claims = compute_fraud_risk_scores(all_flags, claims)
    flags_df = pd.DataFrame(all_flags) if all_flags else pd.DataFrame(
        columns=["claim_id", "flag_type", "severity", "weight", "detail"])
    if verbose:
        print("\n  RISK TIER DISTRIBUTION:")
        tier_counts = scored_claims["risk_tier"].value_counts()
        for tier, count in tier_counts.items():
            pct = count / len(scored_claims) * 100
            print(f"    {tier:<25} {count:>6,} claims  ({pct:.1f}%)")
        print("=" * 65)
    return scored_claims, flags_df

#  REPORTING HELPER

def generate_audit_report(scored_claims: pd.DataFrame, flags_df: pd.DataFrame,
                           output_path: str = "sha_audit_report.csv"):
    """Export flagged claims to CSV for the audit unit."""
    audit_cases = scored_claims[scored_claims["risk_tier"] != "CLEAR"].copy()
    audit_cases = audit_cases.merge(flags_df, on="claim_id", how="left")
    audit_cases.to_csv(output_path, index=False)
    print(f"\n  Audit report saved → {output_path}")
    print(f"  Cases for review: {len(audit_cases):,}")
    return audit_cases
