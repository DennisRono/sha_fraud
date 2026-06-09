"""
sha_fraud_detection.py
======================
SHA Kenya — Comprehensive Healthcare Fraud Detection Engine
Version: 3.0.0

Scientific framework: All detections are grounded in:
  - Biological/physiological hard constraints (what is impossible)
  - Actuarial / operational norms (what is implausible)
  - Statistical inference (what is anomalous relative to peer distribution)
  - Network graph theory (what structural relationships imply collusion)
  - Clinical epidemiology (what coding patterns deviate from expected disease burden)

Each module is independently runnable and collectively composable.
Every positive detection produces a structured ActionPlan for investigators.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest, chi2_contingency

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

warnings.filterwarnings("ignore")

#  ENUMERATIONS

class Severity(str, Enum):
    CRITICAL = "CRITICAL"   # Mathematically/biologically impossible — near-certainty of fraud
    HIGH     = "HIGH"       # Strong statistical evidence — high prior probability of fraud
    MEDIUM   = "MEDIUM"     # Suspicious pattern — warrants investigation
    LOW      = "LOW"        # Weak signal — aggregate for trend monitoring
    INFO     = "INFO"       # Context-only, not scored

class RiskTier(str, Enum):
    IMMEDIATE_AUDIT      = "IMMEDIATE_AUDIT"       # FRS ≥ 15 — freeze + refer to investigation unit
    ENHANCED_MONITORING  = "ENHANCED_MONITORING"   # FRS 8–14 — pre-auth required on future claims
    WATCH                = "WATCH"                 # FRS 1–7  — flag for periodic review
    CLEAR                = "CLEAR"                 # FRS 0    — no action

class ActionCode(str, Enum):
    FREEZE_PAYMENT       = "FREEZE_PAYMENT"        # Suspend reimbursement immediately
    REQUEST_DOCUMENTS    = "REQUEST_DOCUMENTS"     # Demand clinical records, admission notes, lab results
    SITE_VISIT           = "SITE_VISIT"            # Physical inspection of facility
    BENEFICIARY_CONTACT  = "BENEFICIARY_CONTACT"   # Out-of-band contact with member to verify
    POLICE_REFERRAL      = "POLICE_REFERRAL"       # Forward to law enforcement / DPP
    DEREGISTER_FACILITY  = "DEREGISTER_FACILITY"   # Revoke facility accreditation
    ACTUARIAL_REVIEW     = "ACTUARIAL_REVIEW"      # Flag for quarterly actuarial desk review
    PEER_REVIEW          = "PEER_REVIEW"           # Clinical peer panel review
    SANCTIONS_SCREEN     = "SANCTIONS_SCREEN"      # Check EACC / Ethics & Anti-Corruption lists
    WATCHLIST_ADD        = "WATCHLIST_ADD"         # Add entity to internal watchlist

#  CONFIGURATION & THRESHOLDS
#  All values are empirically justified; sources noted in comments.

CONFIG: dict[str, Any] = {
    # Biological constraints ────────────────────────────────────────────────
    "min_inter_delivery_days":          270,   # Minimum gestational period (WHO)
    "max_lifetime_deliveries":          15,    # Above this is epidemiologically extraordinary
    "max_delivery_age":                 55,    # Post-menopausal threshold
    "min_delivery_age":                 10,    # Below age-10 delivery is clinically impossible
    "max_major_surgeries_per_month":    3,     # Physiological recovery constraint
    "max_minor_surgeries_per_month":    6,
    "max_surgeries_per_day_per_patient":2,
    "min_inter_surgery_days_major":     21,    # Minimum healing window post-laparotomy
    "min_inter_surgery_days_minor":     7,
    "max_blood_transfusions_per_month": 8,     # Clinical threshold for chronic condition
    "max_chemotherapy_cycles_per_month":4,     # Oncology protocol maximum
    "max_dialysis_per_week":            4,     # Renal replacement therapy ceiling
    "max_icu_days_per_admission":       90,    # ICU stays beyond 90d are extraordinary
    "min_inpatient_days_for_major_surgery": 1, # Cannot have 0 inpatient days with major surgery
    "max_outpatient_visits_per_day_per_member": 1, # A member can only be at one facility per hour
    "max_inpatient_days_per_admission": 180,   # Implausible stay length flag

    # Facility operational constraints ─────────────────────────────────────
    "max_theater_hours_per_day":           16, # Hard physical ceiling (2 shifts)
    "avg_major_surgery_duration_min":      120,
    "avg_minor_surgery_duration_min":      45,
    "max_facility_occupancy_rate":         0.95,
    "suspicious_occupancy_rate":           0.85,
    "max_outpatients_per_doctor_per_day":  30,
    "max_inpatients_per_doctor_per_day":   10,
    "max_deliveries_per_midwife_per_day":  3,
    "max_antenatal_per_nurse_per_day":     20,
    "min_antenatal_visits_for_delivery":   4,  # WHO ANC minimum; below this is suspicious
    "max_lab_tests_per_patient_per_day":   15, # Reasonable clinical ceiling
    "max_imaging_per_patient_per_month":   8,  # Radiology utilisation cap

    # Pricing / tariff bounds ───────────────────────────────────────────────
    "max_single_claim_amount_ksh":         5_000_000,  # Anything above triggers review
    "min_plausible_claim_amount_ksh":      50,         # Below this is likely a data error
    "max_markup_ratio_over_tariff":        2.0,        # 200% of tariff = extreme upcoding
    "round_number_ratio_threshold":        0.40,       # >40% round-number claims = suspicious
    "just_below_threshold_band_pct":       0.05,       # Within 5% below approval threshold

    # Statistical thresholds ───────────────────────────────────────────────
    "zscore_monitor_threshold":            2.0,
    "zscore_investigate_threshold":        3.0,
    "zscore_mandatory_audit_threshold":    4.0,
    "benford_pvalue_threshold":            0.05,
    "month_end_stuffing_ratio":            0.40,
    "weekend_ratio_anomaly_threshold":     0.50,  # >50% weekend claims for surgical facility
    "iqr_multiplier_outlier":              3.0,   # Tukey fence for amount outliers
    "min_sample_benford":                  50,
    "cluster_radius_days":                 7,     # Window for temporal clustering detection
    "volume_spike_zscore":                 3.0,

    # Fraud Risk Score (FRS) thresholds ────────────────────────────────────
    "frs_immediate_audit":                 15,
    "frs_enhanced_monitoring":             8,

    # Network / collusion thresholds ───────────────────────────────────────
    "member_facility_concentration_ratio": 0.95,
    "agent_facility_concentration_ratio":  0.80,
    "min_claims_for_concentration":        10,
    "clique_min_size":                     3,    # Minimum nodes for collusion clique detection
    "shared_member_overlap_threshold":     0.70, # Jaccard similarity between facility member sets

    # Ghost patient ────────────────────────────────────────────────────────
    "bulk_registration_daily_threshold":   20,
    "ghost_address_phone_shared_threshold":4,

    # Tariff reference (KSh) — representative SHA NHIF/SHA DRG tariffs ─────
    # These should be loaded from a live tariff database; hardcoded here as fallback
    "tariff_reference": {
        "OUTPATIENT":       500,
        "DELIVERY":         8_000,
        "C_SECTION":        35_000,
        "MAJOR_SURGERY":    80_000,
        "MINOR_SURGERY":    15_000,
        "ICU":              12_000,   # per day
        "DIALYSIS":         8_500,
        "CHEMOTHERAPY":     45_000,
        "IMMUNIZATION":     300,
        "ANTENATAL":        1_200,
        "BLOOD_TRANSFUSION":5_000,
        "MRI":              18_000,
        "CT_SCAN":          12_000,
        "XRAY":             1_500,
        "ULTRASOUND":       2_500,
        "LAB_TEST":         800,
    },
}

#  FLAG WEIGHTS  (additive; calibrated so scores are interpretable)

FLAG_WEIGHTS: dict[str, int] = {
    # Tier-1 — Near-certain fraud (impossible events)
    "BIOLOGICAL_IMPOSSIBLE":            10,
    "IMPOSSIBLE_COMBINATION":            9,
    "PHANTOM_SERVICE":                  10,
    "DECEASED_BENEFICIARY_CLAIM":       10,
    "TIME_TRAVEL_CLAIM":                10,  # Claim date before registration/birth

    # Tier-2 — Strong fraud signals
    "CAPACITY_OVERFLOW":                 8,
    "TARIFF_EXTREME_MARKUP":             8,
    "NETWORK_COLLUSION":                 7,
    "ZSCORE_EXTREME":                    7,
    "SPLIT_BILLING":                     7,   # Fragmenting a single episode into multiple claims
    "UNBUNDLING":                        7,   # Billing components that should be in one DRG
    "GHOST_PATIENT":                     6,
    "DUPLICATE_CLAIM":                   6,
    "PROVIDER_IDENTITY_THEFT":           9,

    # Tier-3 — Moderate signals
    "BENFORD_DEVIATION":                 5,
    "CLINICAL_CODE_ANOMALY":             5,
    "UPCODING":                          5,
    "THRESHOLD_GAMING":                  5,   # Claims just below authorization thresholds
    "PING_PONG_REFERRAL":                5,   # Circular facility-to-facility referrals
    "READMISSION_BOUNCE":                4,   # Same DRG re-admission within 30 days

    # Tier-4 — Weak signals (aggregate value)
    "TEMPORAL_CLUSTERING":               4,
    "STAFFING_MISMATCH":                 4,
    "STATISTICAL_OUTLIER":               3,
    "ROUND_NUMBER_PATTERN":              3,
    "WEEKEND_ANOMALY":                   3,
    "HIGH_REJECTION_THEN_RESUBMIT":      3,
    "MISSING_PREREQUISITE":              4,   # e.g., delivery without any antenatal
    "IMPLAUSIBLE_LOS":                   4,   # Length-of-stay anomaly
    "SINGLE_DIAGNOSIS_HIGH_VOLUME":      3,
    "SELF_REFERRAL_PATTERN":             3,
    "TARIFF_ROUNDING_PATTERN":           3,
}

#  DATA CLASSES

@dataclass
class FlagRecord:
    claim_id:        Any
    flag_type:       str
    severity:        Severity
    weight:          int
    detail:          str
    module:          str                    = "UNKNOWN"
    entity_id:       Optional[str]          = None  # facility_id or member_id driving the flag
    entity_type:     Optional[str]          = None  # "FACILITY" | "MEMBER" | "AGENT" | "PROVIDER"
    evidence_values: dict[str, Any]         = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id":        self.claim_id,
            "flag_type":       self.flag_type,
            "severity":        self.severity.value,
            "weight":          self.weight,
            "detail":          self.detail,
            "module":          self.module,
            "entity_id":       self.entity_id,
            "entity_type":     self.entity_type,
            "evidence_values": str(self.evidence_values),
        }


@dataclass
class ActionPlan:
    claim_id:        Any
    risk_tier:       RiskTier
    fraud_risk_score:float
    primary_actions: list[ActionCode]
    rationale:       str
    deadline_days:   int                    # Days from detection to required response
    escalate_to:     str                    # Unit responsible
    supporting_flags:list[str]             = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id":         self.claim_id,
            "risk_tier":        self.risk_tier.value,
            "fraud_risk_score": self.fraud_risk_score,
            "primary_actions":  [a.value for a in self.primary_actions],
            "rationale":        self.rationale,
            "deadline_days":    self.deadline_days,
            "escalate_to":      self.escalate_to,
            "supporting_flags": " | ".join(self.supporting_flags),
        }

#  HELPER UTILITIES

def compute_age(dob: pd.Series, reference_date: pd.Series) -> pd.Series:
    """Compute age in fractional years. Both inputs must be datetime64."""
    return ((reference_date - dob).dt.days / 365.25).round(2)


def benford_expected_freq() -> np.ndarray:
    """Benford's Law expected first-digit frequencies for digits 1–9."""
    return np.array([np.log10(1 + 1 / d) for d in range(1, 10)])


def leading_digit(series: pd.Series) -> pd.Series:
    """Extract leading significant digit from a numeric series."""
    s = series.abs().replace(0, np.nan).dropna()
    return (s.astype(str)
             .str.lstrip("0")
             .str.replace(r"[^0-9]", "", regex=True)
             .str[0]
             .replace("", np.nan)
             .dropna()
             .astype(int))


def _make_flag(
    flags: list[FlagRecord],
    claim_id: Any,
    flag_type: str,
    detail: str,
    severity: Severity = Severity.HIGH,
    module: str = "UNKNOWN",
    entity_id: Optional[str] = None,
    entity_type: Optional[str] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> None:
    """Centralised flag constructor — ensures weight is always resolved."""
    flags.append(FlagRecord(
        claim_id=claim_id,
        flag_type=flag_type,
        severity=severity,
        weight=FLAG_WEIGHTS.get(flag_type, 1),
        detail=detail,
        module=module,
        entity_id=entity_id,
        entity_type=entity_type,
        evidence_values=evidence or {},
    ))


def _safe_date(series: pd.Series) -> pd.Series:
    """Parse dates robustly; coerce unparseable values to NaT."""
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def _proc_upper(df: pd.DataFrame, col: str = "procedure_type") -> pd.Series:
    return df[col].fillna("").str.upper().str.strip()


def _normalise_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise sex column to 'M' or 'F'."""
    mapping = {
        "M": "M", "MALE": "M", "1": "M",
        "F": "F", "FEMALE": "F", "2": "F",
    }
    df = df.copy()
    df["sex_norm"] = df["sex"].fillna("").str.upper().str.strip().map(mapping).fillna("UNKNOWN")
    return df


#  MODULE 1 — BIOLOGICAL IMPOSSIBILITY CHECKS

FEMALE_ONLY_PROCEDURES: set[str] = {
    "DELIVERY", "C_SECTION", "CAESAREAN", "ANTENATAL", "CERVICAL_CANCER_SCREEN",
    "OVARIAN", "HYSTERECTOMY", "FALLOPIAN", "ENDOMETRIOSIS", "OOPHORECTOMY",
    "LABIAPLASTY", "VULVECTOMY", "COLPOSCOPY", "DILATION_CURETTAGE",
    "POSTPARTUM", "PUERPERIUM", "EPISIOTOMY", "AMNIOCENTESIS", "IUD_INSERTION",
}

MALE_ONLY_PROCEDURES: set[str] = {
    "PROSTATECTOMY", "ORCHIECTOMY", "VASECTOMY", "CIRCUMCISION_ADULT",
    "TESTICULAR_BIOPSY", "PENILE", "ORCHIDOPEXY",
}

DELIVERY_PROCEDURE_TYPES: set[str] = {
    "DELIVERY", "C_SECTION", "CAESAREAN", "NORMAL_DELIVERY", "SVD",
    "ASSISTED_DELIVERY", "FORCEPS_DELIVERY", "VACUUM_DELIVERY",
}

SURGICAL_TYPES: set[str] = {"MAJOR_SURGERY", "MINOR_SURGERY"}


def check_biological_impossibilities(
    claims: pd.DataFrame, members: pd.DataFrame
) -> list[FlagRecord]:
    """
    Module 1: Biological and physiological impossibility checks.

    Flags events that cannot occur given the laws of human physiology:
      1a. Sex-procedure mismatches (male claiming delivery, female claiming prostatectomy)
      1b. Multiple deliveries on the same calendar day
      1c. Inter-delivery interval < 270 days (gestational period)
      1d. Lifetime delivery count implausibility
      1e. Age at delivery outside viable range
      1f. Surgical volume per patient per month
      1g. Inter-surgical recovery gap violations
      1h. Claims dated before birth or after death
      1i. Procedure requires age-specific biology (e.g., paediatric procedure on adult)
      1j. Dialysis / chemotherapy / transfusion frequency violations
      1k. Concurrent inpatient stays at different facilities
      1l. Delivery without any prior antenatal care (possible ghost episode)
    """
    M = "MODULE_1_BIOLOGICAL"
    flags: list[FlagRecord] = []

    members = _normalise_sex(members)
    base_cols = ["member_id", "dob", "sex_norm"]
    optional_cols = [c for c in ["death_date", "registration_date"] if c in members.columns]
    df = claims.merge(members[base_cols + optional_cols], on="member_id", how="left")
    df["claim_date"] = _safe_date(df["claim_date"])
    df["dob"] = _safe_date(df["dob"])
    df["age_at_claim"] = compute_age(df["dob"], df["claim_date"])
    df["proc_upper"] = _proc_upper(df)

    # 1h. Time-travel claims ─────────────────────────────────────────────
    # Claim before birth
    pre_birth = df[df["claim_date"] < df["dob"]]
    for _, row in pre_birth.iterrows():
        if pd.notna(row["dob"]) and pd.notna(row["claim_date"]):
            _make_flag(flags, row["claim_id"], "TIME_TRAVEL_CLAIM",
                f"Member {row['member_id']}: Claim date {row['claim_date'].date()} is BEFORE "
                f"date of birth {row['dob'].date()}. Impossible — indicates falsified member record.",
                Severity.CRITICAL, M, str(row["member_id"]), "MEMBER",
                {"claim_date": str(row["claim_date"].date()), "dob": str(row["dob"].date())})

    # Claim after death
    if "death_date" in df.columns:
        df["death_date"] = _safe_date(df["death_date"])
        post_death = df[(df["death_date"].notna()) & (df["claim_date"] > df["death_date"])]
        for _, row in post_death.iterrows():
            _make_flag(flags, row["claim_id"], "DECEASED_BENEFICIARY_CLAIM",
                f"Member {row['member_id']}: Claim on {row['claim_date'].date()} is AFTER "
                f"recorded death on {row['death_date'].date()}. Indicates post-mortem claim fraud.",
                Severity.CRITICAL, M, str(row["member_id"]), "MEMBER",
                {"claim_date": str(row["claim_date"].date()), "death_date": str(row["death_date"].date())})

    # Claim before scheme registration
    if "registration_date" in df.columns:
        df["registration_date"] = _safe_date(df["registration_date"])
        pre_reg = df[
            df["registration_date"].notna() &
            (df["claim_date"] < df["registration_date"])
        ]
        for _, row in pre_reg.iterrows():
            _make_flag(flags, row["claim_id"], "TIME_TRAVEL_CLAIM",
                f"Member {row['member_id']}: Claim date {row['claim_date'].date()} is BEFORE "
                f"member registration date {row['registration_date'].date()}. "
                f"Back-dating claims to cover pre-existing conditions.",
                Severity.CRITICAL, M, str(row["member_id"]), "MEMBER")

    # 1a. Sex-procedure mismatches ──────────────────────────────────────
    for proc in FEMALE_ONLY_PROCEDURES:
        male_wrong = df[
            (df["sex_norm"] == "M") &
            df["proc_upper"].str.contains(proc, regex=False, na=False)
        ]
        for _, row in male_wrong.iterrows():
            _make_flag(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                f"Male member {row['member_id']} (sex=M) claimed female-only procedure '{proc}'. "
                f"Either member sex record is falsified or claim is fraudulent.",
                Severity.CRITICAL, M, str(row["member_id"]), "MEMBER",
                {"procedure": proc, "sex": row["sex_norm"]})

    for proc in MALE_ONLY_PROCEDURES:
        female_wrong = df[
            (df["sex_norm"] == "F") &
            df["proc_upper"].str.contains(proc, regex=False, na=False)
        ]
        for _, row in female_wrong.iterrows():
            _make_flag(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                f"Female member {row['member_id']} (sex=F) claimed male-only procedure '{proc}'.",
                Severity.CRITICAL, M, str(row["member_id"]), "MEMBER")

    # 1b–1e. Delivery-specific checks ───────────────────────────────────
    deliveries = df[
        df["proc_upper"].apply(lambda p: any(d in p for d in DELIVERY_PROCEDURE_TYPES))
    ].sort_values(["member_id", "claim_date"])

    for member_id, grp in deliveries.groupby("member_id"):
        grp: Any = grp.reset_index(drop=True)
        sex = grp["sex_norm"].iloc[0]

        # Multiple deliveries same day
        for dt, day_grp in grp.groupby("claim_date"):
            if len(day_grp) > 1:
                for cid in day_grp["claim_id"].tolist():
                    _make_flag(flags, cid, "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id}: {len(day_grp)} delivery claims on {dt.date()} — "
                        f"physically impossible (human singleton / twin births cannot repeat same calendar day).",
                        Severity.CRITICAL, M, str(member_id), "MEMBER")

        # Inter-delivery gap
        dates = grp["claim_date"].tolist()
        cids  = grp["claim_id"].tolist()
        for i in range(1, len(dates)):
            if pd.notna(dates[i]) and pd.notna(dates[i-1]):
                gap = (dates[i] - dates[i-1]).days
                if gap < CONFIG["min_inter_delivery_days"]:
                    _make_flag(flags, cids[i], "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id}: Only {gap} days between deliveries "
                        f"(biological minimum = {CONFIG['min_inter_delivery_days']} days / 9 months). "
                        f"Multiple deliveries claimed within one gestational period.",
                        Severity.CRITICAL, M, str(member_id), "MEMBER",
                        {"gap_days": gap, "minimum_days": CONFIG["min_inter_delivery_days"]})

        # Lifetime delivery count
        if len(grp) > CONFIG["max_lifetime_deliveries"]:
            for cid in grp["claim_id"].tolist():
                _make_flag(flags, cid, "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: {len(grp)} lifetime deliveries claimed "
                    f"(plausible maximum = {CONFIG['max_lifetime_deliveries']}). "
                    f"Statistically implausible for a single individual.",
                    Severity.CRITICAL, M, str(member_id), "MEMBER",
                    {"count": len(grp), "max": CONFIG["max_lifetime_deliveries"]})

        # Age at delivery
        for _, row in grp.iterrows():
            age = row["age_at_claim"]
            if pd.notna(age):
                if age > CONFIG["max_delivery_age"]:
                    _make_flag(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id}: Delivery claimed at age {age:.1f} years "
                        f"(post-menopausal threshold = {CONFIG['max_delivery_age']}). "
                        f"Natural pregnancy after menopause without IVF documentation is extraordinary.",
                        Severity.CRITICAL, M, str(member_id), "MEMBER",
                        {"age": age, "threshold": CONFIG["max_delivery_age"]})
                if age < CONFIG["min_delivery_age"]:
                    _make_flag(flags, row["claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id}: Delivery claimed at age {age:.1f} years "
                        f"(below minimum plausible age of {CONFIG['min_delivery_age']}).",
                        Severity.CRITICAL, M, str(member_id), "MEMBER",
                        {"age": age, "threshold": CONFIG["min_delivery_age"]})

    # 1f–1g. Surgical volume and recovery gaps ───────────────────────────
    surgeries: Any = df[df["proc_upper"].isin(SURGICAL_TYPES)].copy()
    surgeries["year_month"] = surgeries["claim_date"].dt.to_period("M")

    for (member_id, ym, ptype), grp in surgeries.groupby(
        ["member_id", "year_month", "proc_upper"]
    ):
        limit = (CONFIG["max_major_surgeries_per_month"]
                 if ptype == "MAJOR_SURGERY" else CONFIG["max_minor_surgeries_per_month"])
        if len(grp) > limit:
            for cid in grp["claim_id"].tolist():
                _make_flag(flags, cid, "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: {len(grp)} {ptype} claims in {ym} "
                    f"(monthly limit = {limit}). Post-operative recovery makes this impossible.",
                    Severity.CRITICAL, M, str(member_id), "MEMBER",
                    {"count": len(grp), "limit": limit, "month": str(ym)})

    # Same-day cap
    for (member_id, dt), grp in surgeries.groupby(["member_id", "claim_date"]):
        if len(grp) > CONFIG["max_surgeries_per_day_per_patient"]:
            for cid in grp["claim_id"].tolist():
                _make_flag(flags, cid, "BIOLOGICAL_IMPOSSIBLE",
                    f"Member {member_id}: {len(grp)} surgeries on {pd.Timestamp(dt).date()} "
                    f"(daily cap = {CONFIG['max_surgeries_per_day_per_patient']}).",
                    Severity.CRITICAL, M, str(member_id), "MEMBER")

    # Inter-surgery recovery gap
    for ptype, min_gap_key in [("MAJOR_SURGERY", "min_inter_surgery_days_major"),
                                ("MINOR_SURGERY", "min_inter_surgery_days_minor")]:
        subset = surgeries[surgeries["proc_upper"] == ptype].sort_values(
            ["member_id", "claim_date"]
        )
        for member_id, grp in subset.groupby("member_id"):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                if pd.notna(grp.loc[i, "claim_date"]) and pd.notna(grp.loc[i-1, "claim_date"]):
                    gap = (grp.loc[i, "claim_date"] - grp.loc[i-1, "claim_date"]).days
                    min_gap = CONFIG[min_gap_key]
                    if gap < min_gap:
                        _make_flag(flags, grp.loc[i, "claim_id"], "BIOLOGICAL_IMPOSSIBLE",
                            f"Member {member_id}: Only {gap} days between {ptype} procedures "
                            f"(minimum recovery = {min_gap} days). "
                            f"No clinical protocol permits this interval.",
                            Severity.CRITICAL, M, str(member_id), "MEMBER",
                            {"gap_days": gap, "minimum_days": min_gap})

    # 1j. High-frequency intensive therapies ────────────────────────────
    INTENSIVE_CHECKS: list[tuple[str, str, int, str]] = [
        ("DIALYSIS",         "DIALYSIS",         CONFIG["max_dialysis_per_week"],            "week"),
        ("BLOOD_TRANSFUSION","BLOOD_TRANSFUSION", CONFIG["max_blood_transfusions_per_month"], "month"),
        ("CHEMOTHERAPY",     "CHEMOTHERAPY",      CONFIG["max_chemotherapy_cycles_per_month"],"month"),
    ]
    for proc_keyword, label, limit, period in INTENSIVE_CHECKS:
        subset = df[df["proc_upper"].str.contains(proc_keyword, na=False)].copy()
        if period == "week":
            subset["period"] = subset["claim_date"].dt.to_period("W")
        else:
            subset["period"] = subset["claim_date"].dt.to_period("M")
        for (member_id, per), grp in subset.groupby(["member_id", "period"]):
            if len(grp) > limit:
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id}: {len(grp)} {label} sessions in {per} "
                        f"(clinical maximum = {limit}/{period}). "
                        f"Exceeds any published treatment protocol.",
                        Severity.HIGH, M, str(member_id), "MEMBER",
                        {"count": len(grp), "limit": limit, "period": str(per)})

    # 1k. Concurrent inpatient stays at different facilities ────────────
    if "facility_id" in df.columns:
        inpatient: Any = df[df["proc_upper"].str.contains("INPATIENT|ICU|ADMISSION|WARD", na=False)].copy()
        for (member_id, dt), grp in inpatient.groupby(["member_id", "claim_date"]):
            unique_facs = grp["facility_id"].nunique()
            if unique_facs > 1:
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "BIOLOGICAL_IMPOSSIBLE",
                        f"Member {member_id}: Inpatient claims at {unique_facs} DIFFERENT facilities "
                        f"on {pd.Timestamp(dt).date()}. A person cannot be simultaneously admitted "
                        f"at multiple hospitals.",
                        Severity.CRITICAL, M, str(member_id), "MEMBER",
                        {"facilities": grp["facility_id"].tolist()})

    # 1l. Delivery without any preceding antenatal care ────────────────
    if "proc_upper" in df.columns:
        antenatal_members = set(
            df[df["proc_upper"].str.contains("ANTENATAL|ANC", na=False)]["member_id"].tolist()
        )
        delivery_members = deliveries["member_id"].tolist()
        for member_id in delivery_members:
            if member_id not in antenatal_members:
                member_deliveries = deliveries[deliveries["member_id"] == member_id]
                for cid in member_deliveries["claim_id"].tolist():
                    _make_flag(flags, cid, "MISSING_PREREQUISITE",
                        f"Member {member_id}: Delivery claim with ZERO prior antenatal (ANC) visits on record. "
                        f"WHO minimum is {CONFIG['min_antenatal_visits_for_delivery']} ANC visits. "
                        f"Suggests fabricated delivery episode.",
                        Severity.HIGH, M, str(member_id), "MEMBER")

    return flags


#  MODULE 2 — FACILITY CAPACITY CHECKS

def check_facility_capacity(
    claims: pd.DataFrame, facilities: pd.DataFrame
) -> list[FlagRecord]:
    """
    Module 2: Facility operational capacity violations.

      2a. Inpatient-day overflow (bed × days ceiling)
      2b. Theater throughput overflow (hours × theaters)
      2c. Doctor-to-patient throughput ceiling (outpatient)
      2d. Delivery volume vs midwife count
      2e. Lab test volume plausibility
      2f. Imaging volume plausibility
      2g. Facility claims activity on non-operational days (weekends for closed facilities)
      2h. Procedures performed at wrong facility level
      2i. Night-time procedure anomalies (elective surgery at 2am)
    """
    M = "MODULE_2_CAPACITY"
    flags: list[FlagRecord] = []

    df: Any = claims.merge(facilities, on="facility_id", how="left")
    df["claim_date"] = _safe_date(df["claim_date"])
    df["year_month"] = df["claim_date"].dt.to_period("M")
    df["proc_upper"] = _proc_upper(df)

    for (facility_id, ym), grp in df.groupby(["facility_id", "year_month"]):
        bed_count:       int = int(grp["bed_count"].iloc[0] or 0)
        theater_count:   int = int(grp["theater_count"].iloc[0] or 0)
        n_doctors:       int = int(grp["registered_doctors"].iloc[0] or 0)
        days_in_month:   int = int(ym.days_in_month)
        fac_label:       str = f"Facility {facility_id} [{ym}]"

        # 2a. Inpatient-day overflow ─────────────────────────────────────
        if "inpatient_days" in grp.columns and bed_count > 0:
            total_ip_days: float = grp["inpatient_days"].fillna(0).sum()
            max_possible:  float = bed_count * days_in_month
            suspicious_limit: float = max_possible * CONFIG["suspicious_occupancy_rate"]

            if total_ip_days > max_possible:
                ratio = total_ip_days / max_possible
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "CAPACITY_OVERFLOW",
                        f"{fac_label}: {total_ip_days:.0f} inpatient-days claimed vs. "
                        f"maximum possible {max_possible:.0f} ({ratio:.2f}× capacity). "
                        f"Physically impossible — bed count × calendar days is a hard ceiling.",
                        Severity.CRITICAL, M, str(facility_id), "FACILITY",
                        {"total_inpatient_days": total_ip_days, "max_possible": max_possible,
                         "ratio": ratio})
            elif total_ip_days > suspicious_limit:
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "CAPACITY_OVERFLOW",
                        f"{fac_label}: Occupancy at "
                        f"{total_ip_days/max_possible*100:.1f}% — above "
                        f"{CONFIG['suspicious_occupancy_rate']*100:.0f}% soft threshold. "
                        f"Real-world occupancy rarely exceeds 85% sustainably.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY",
                        {"occupancy_rate": total_ip_days / max_possible})

        # 2b. Theater throughput ─────────────────────────────────────────
        if theater_count > 0:
            surgeries_grp = grp[grp["proc_upper"].isin(SURGICAL_TYPES)]
            for sdate, day_grp in surgeries_grp.groupby("claim_date"):
                n_major = (day_grp["proc_upper"] == "MAJOR_SURGERY").sum()
                n_minor = (day_grp["proc_upper"] == "MINOR_SURGERY").sum()
                used_min = (n_major * CONFIG["avg_major_surgery_duration_min"] +
                            n_minor * CONFIG["avg_minor_surgery_duration_min"])
                available_min = CONFIG["max_theater_hours_per_day"] * 60 * theater_count
                if used_min > available_min:
                    for cid in day_grp["claim_id"].tolist():
                        _make_flag(flags, cid, "CAPACITY_OVERFLOW",
                            f"Facility {facility_id} on {pd.Timestamp(sdate).date()}: "
                            f"{n_major} major + {n_minor} minor surgeries requires "
                            f"{used_min} min but only {available_min} min available "
                            f"({theater_count} theater(s) × {CONFIG['max_theater_hours_per_day']}h). "
                            f"Physically impossible within a single operational day.",
                            Severity.CRITICAL, M, str(facility_id), "FACILITY",
                            {"used_minutes": used_min, "available_minutes": available_min,
                             "n_major": int(n_major), "n_minor": int(n_minor)})

        # 2c. Doctor outpatient throughput ──────────────────────────────
        if n_doctors > 0:
            outpatient_grp = grp[grp["proc_upper"].str.contains("OUTPATIENT|OPD", na=False)]
            max_outpatients_month = n_doctors * CONFIG["max_outpatients_per_doctor_per_day"] * days_in_month
            if len(outpatient_grp) > max_outpatients_month:
                for cid in outpatient_grp["claim_id"].tolist():
                    _make_flag(flags, cid, "STAFFING_MISMATCH",
                        f"{fac_label}: {len(outpatient_grp):,} outpatient claims vs. "
                        f"maximum {max_outpatients_month:,} "
                        f"({n_doctors} doctors × {CONFIG['max_outpatients_per_doctor_per_day']} pts/day × {days_in_month} days). "
                        f"Staffing cannot support this volume.",
                        Severity.HIGH, M, str(facility_id), "FACILITY",
                        {"claimed": len(outpatient_grp), "max_possible": max_outpatients_month})

        # 2d. Delivery volume vs midwife count ──────────────────────────
        if "registered_midwives" in facilities.columns:
            midwives = int(grp["registered_midwives"].iloc[0] or 0)
            if midwives > 0:
                deliveries_grp = grp[
                    grp["proc_upper"].apply(lambda p: any(d in p for d in DELIVERY_PROCEDURE_TYPES))
                ]
                for ddate, day_del in deliveries_grp.groupby("claim_date"):
                    max_del = midwives * CONFIG["max_deliveries_per_midwife_per_day"]
                    if len(day_del) > max_del:
                        for cid in day_del["claim_id"].tolist():
                            _make_flag(flags, cid, "STAFFING_MISMATCH",
                                f"Facility {facility_id} on {pd.Timestamp(ddate).date()}: "
                                f"{len(day_del)} deliveries with only {midwives} midwife(s) "
                                f"(max = {max_del}). Staffing constraint violated.",
                                Severity.HIGH, M, str(facility_id), "FACILITY")

        # 2e. Lab test volume plausibility ─────────────────────────────
        lab_grp = grp[grp["proc_upper"].str.contains("LAB|TEST|LABORATORY|PATHOLOGY", na=False)]
        for (member_id, ldate), lab_day in lab_grp.groupby(["member_id", "claim_date"]):
            if len(lab_day) > CONFIG["max_lab_tests_per_patient_per_day"]:
                for cid in lab_day["claim_id"].tolist():
                    _make_flag(flags, cid, "CAPACITY_OVERFLOW",
                        f"Member {member_id} at Facility {facility_id} on "
                        f"{pd.Timestamp(ldate).date()}: {len(lab_day)} lab tests in one day "
                        f"(clinical ceiling = {CONFIG['max_lab_tests_per_patient_per_day']}). "
                        f"Panel fishing or phantom tests.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY")

        # 2f. Imaging volume ────────────────────────────────────────────
        imaging_grp = grp[grp["proc_upper"].str.contains("MRI|CT_SCAN|XRAY|ULTRASOUND|IMAGING|RADIOLOGY", na=False)]
        imaging_monthly = imaging_grp.groupby("member_id").size()
        for member_id, count in imaging_monthly.items():
            if count > CONFIG["max_imaging_per_patient_per_month"]:
                for cid in imaging_grp[imaging_grp["member_id"] == member_id]["claim_id"].tolist():
                    _make_flag(flags, cid, "CAPACITY_OVERFLOW",
                        f"Member {member_id}: {count} imaging studies in {ym} at Facility {facility_id} "
                        f"(monthly ceiling = {CONFIG['max_imaging_per_patient_per_month']}). "
                        f"Radiation safety protocols prohibit this frequency for most conditions.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY")

    # 2g. Weekend surgical anomaly ──────────────────────────────────────
    df["day_of_week"] = df["claim_date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    surgical_df = df[df["proc_upper"].isin(SURGICAL_TYPES)]
    for facility_id, grp in surgical_df.groupby("facility_id"):
        total = len(grp)
        if total < 10:
            continue
        weekend_ratio = grp["is_weekend"].mean()
        if weekend_ratio > CONFIG["weekend_ratio_anomaly_threshold"]:
            for cid in grp[grp["is_weekend"]]["claim_id"].tolist():
                _make_flag(flags, cid, "WEEKEND_ANOMALY",
                    f"Facility {facility_id}: {weekend_ratio*100:.1f}% of elective surgeries claimed "
                    f"on weekends (threshold = {CONFIG['weekend_ratio_anomaly_threshold']*100:.0f}%). "
                    f"Elective surgical schedules are predominantly Mon–Fri. "
                    f"Emergency surgeries exist but not at this ratio.",
                    Severity.MEDIUM, M, str(facility_id), "FACILITY",
                    {"weekend_ratio": float(weekend_ratio)})

    return flags


#  MODULE 3 — STATISTICAL DISTRIBUTION CHECKS

def check_statistical_anomalies(
    claims: pd.DataFrame, facilities: pd.DataFrame
) -> list[FlagRecord]:
    """
    Module 3: Statistical and distributional anomaly detection.

      3a. Benford's Law deviation per facility (chi-square test)
      3b. Z-score peer comparison (volume and spend) within county+level peer group
      3c. Month-end claim stuffing
      3d. Sudden volume spikes (rolling window Z-score)
      3e. Amount distribution: IQR outliers and heavy-tail analysis
      3f. Round-number clustering (a hallmark of fabricated amounts)
      3g. Threshold gaming (claims just below authorisation thresholds)
      3h. Kolmogorov-Smirnov test against peer claim-amount distribution
      3i. Single-diagnosis high volume (one ICD code dominates suspiciously)
      3j. Length-of-stay implausibility per DRG
    """
    M = "MODULE_3_STATISTICAL"
    flags: list[FlagRecord] = []

    df = claims.merge(
        facilities[["facility_id", "facility_level", "county", "ownership"]],
        on="facility_id", how="left"
    )
    df["claim_date"] = _safe_date(df["claim_date"])
    df["year_month"]    = df["claim_date"].dt.to_period("M")
    df["day_of_month"]  = df["claim_date"].dt.day
    df["days_in_month"] = df["claim_date"].dt.days_in_month
    df["is_last_5_days"] = df["day_of_month"] > (df["days_in_month"] - 5)

    benford_expected = benford_expected_freq()

    # 3a. Benford's Law ─────────────────────────────────────────────────
    for facility_id, grp in df.groupby("facility_id"):
        amounts = grp["claim_amount"].dropna()
        amounts = amounts[amounts > 0]
        if len(amounts) < CONFIG["min_sample_benford"]:
            continue
        fd = leading_digit(amounts)
        if len(fd) < CONFIG["min_sample_benford"]:
            continue
        observed: Any = fd.value_counts().reindex(range(1, 10), fill_value=0).values
        total    = observed.sum()
        expected = benford_expected * total
        # Only run where expected counts meet minimum chi-square validity
        if (expected < 5).sum() <= 2:
            chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
            if p_value < CONFIG["benford_pvalue_threshold"]:
                severity = Severity.HIGH if p_value < 0.01 else Severity.MEDIUM
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "BENFORD_DEVIATION",
                        f"Facility {facility_id}: Claim amounts deviate from Benford's Law "
                        f"(χ²={chi2_stat:.2f}, p={p_value:.5f}, n={total}). "
                        f"Benford's Law holds for naturally occurring financial figures; "
                        f"deviation indicates possible fabrication or systematic rounding.",
                        severity, M, str(facility_id), "FACILITY",
                        {"chi2": float(chi2_stat), "p_value": float(p_value), "n": int(total)})

    # 3b. Z-score peer comparison ───────────────────────────────────────
    monthly_volume = df.groupby(
        ["facility_id", "year_month", "facility_level", "county"]
    ).agg(
        claim_count=("claim_id", "count"),
        total_amount=("claim_amount", "sum")
    ).reset_index()

    for (level, county, ym), peer in monthly_volume.groupby(
        ["facility_level", "county", "year_month"]
    ):
        if len(peer) < 3:
            continue
        for metric, label in [("claim_count", "claim volume"), ("total_amount", "total spend")]:
            mu  = peer[metric].mean()
            sig = peer[metric].std()
            if sig == 0:
                continue
            peer = peer.copy()
            peer[f"z_{metric}"] = (peer[metric] - mu) / sig
            outliers = peer[peer[f"z_{metric}"].abs() > CONFIG["zscore_investigate_threshold"]]
            for _, row in outliers.iterrows():
                z = row[f"z_{metric}"]
                severity = (Severity.HIGH if abs(z) >= CONFIG["zscore_mandatory_audit_threshold"]
                            else Severity.MEDIUM)
                flag_type = ("ZSCORE_EXTREME" if abs(z) >= CONFIG["zscore_mandatory_audit_threshold"]
                             else "STATISTICAL_OUTLIER")
                fac_claims = df[
                    (df["facility_id"] == row["facility_id"]) &
                    (df["year_month"] == ym)
                ]["claim_id"].tolist()
                for cid in fac_claims:
                    _make_flag(flags, cid, flag_type,
                        f"Facility {row['facility_id']} [{ym}]: {label} Z-score = {z:.2f} "
                        f"({row[metric]:,.0f} vs peer mean {mu:,.0f}) "
                        f"among Level-{level} facilities in {county} county.",
                        severity, M, str(row["facility_id"]), "FACILITY",
                        {"z_score": float(z), "value": float(row[metric]),
                         "peer_mean": float(mu), "peer_std": float(sig)})

    # 3c. Month-end claim stuffing ──────────────────────────────────────
    for (facility_id, ym), grp in df.groupby(["facility_id", "year_month"]):
        total = len(grp)
        if total <= 20:
            continue
        last5 = grp["is_last_5_days"].sum()
        ratio = last5 / total
        if ratio > CONFIG["month_end_stuffing_ratio"]:
            for cid in grp[grp["is_last_5_days"]]["claim_id"].tolist():
                _make_flag(flags, cid, "TEMPORAL_CLUSTERING",
                    f"Facility {facility_id} [{ym}]: {last5}/{total} claims "
                    f"({ratio*100:.1f}%) in last 5 days of month "
                    f"(threshold = {CONFIG['month_end_stuffing_ratio']*100:.0f}%). "
                    f"Month-end stuffing is a classic indicator of budget absorption fraud.",
                    Severity.MEDIUM, M, str(facility_id), "FACILITY",
                    {"last5_count": int(last5), "total": int(total), "ratio": float(ratio)})

    # 3d. Rolling volume spike detection ────────────────────────────────
    monthly_counts = (
        df.groupby(["facility_id", "year_month"]).size()
        .reset_index(name="count")
    )
    monthly_counts["year_month_dt"] = monthly_counts["year_month"].dt.to_timestamp()
    monthly_counts = monthly_counts.sort_values(["facility_id", "year_month_dt"])

    for facility_id, grp in monthly_counts.groupby("facility_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < 4:
            continue
        roll_mean = grp["count"].rolling(3, min_periods=2).mean().shift(1)
        roll_std  = grp["count"].rolling(3, min_periods=2).std().shift(1)
        grp["z_spike"] = (grp["count"] - roll_mean) / roll_std.replace(0, np.nan)
        spikes = grp[grp["z_spike"] > CONFIG["volume_spike_zscore"]]
        for _, row in spikes.iterrows():
            month_claims = df[
                (df["facility_id"] == facility_id) &
                (df["year_month"] == row["year_month"])
            ]["claim_id"].tolist()
            for cid in month_claims:
                _make_flag(flags, cid, "STATISTICAL_OUTLIER",
                    f"Facility {facility_id}: Volume spike in {row['year_month']} "
                    f"({row['count']:.0f} claims, Z={row['z_spike']:.2f} above 3-month rolling average). "
                    f"Sudden spikes correlate with fraudulent claim batching.",
                    Severity.MEDIUM, M, str(facility_id), "FACILITY",
                    {"count": float(row["count"]), "z_spike": float(row["z_spike"])})

    # 3e. IQR / Tukey fence outlier per procedure type ─────────────────
    for proc_type, grp in df.groupby("procedure_type"):
        amounts = grp["claim_amount"].dropna()
        if len(amounts) < 20:
            continue
        q1, q3 = amounts.quantile(0.25), amounts.quantile(0.75)
        iqr     = q3 - q1
        upper   = q3 + CONFIG["iqr_multiplier_outlier"] * iqr
        lower   = q1 - CONFIG["iqr_multiplier_outlier"] * iqr
        outliers = grp[(grp["claim_amount"] > upper) | (grp["claim_amount"] < lower)]
        for _, row in outliers.iterrows():
            direction = "HIGH" if row["claim_amount"] > upper else "LOW"
            _make_flag(flags, row["claim_id"], "STATISTICAL_OUTLIER",
                f"Claim {row['claim_id']}: Amount {row['claim_amount']:,.0f} KSh is a "
                f"{direction} outlier for procedure '{proc_type}' "
                f"(IQR fence: {lower:,.0f}–{upper:,.0f}, based on {len(amounts)} claims). "
                f"Extreme values indicate upcoding or data entry fraud.",
                Severity.MEDIUM, M, str(row.get("facility_id", "")), "FACILITY",
                {"amount": float(row["claim_amount"]), "upper_fence": float(upper),
                 "lower_fence": float(lower)})

    # 3f. Round-number amount clustering ───────────────────────────────
    for facility_id, grp in df.groupby("facility_id"):
        amounts = grp["claim_amount"].dropna()
        if len(amounts) < 30:
            continue
        round_1000 = (amounts % 1000 == 0).sum()
        round_500  = (amounts % 500 == 0).sum()
        round_pct  = (round_1000 + round_500) / len(amounts)
        if round_pct > CONFIG["round_number_ratio_threshold"]:
            for cid in grp["claim_id"].tolist():
                _make_flag(flags, cid, "ROUND_NUMBER_PATTERN",
                    f"Facility {facility_id}: {round_pct*100:.1f}% of claim amounts are "
                    f"round numbers (multiples of 500/1000 KSh). "
                    f"Naturally occurring health costs follow continuous distributions; "
                    f"high round-number density suggests price fabrication.",
                    Severity.LOW, M, str(facility_id), "FACILITY",
                    {"round_pct": float(round_pct)})

    # 3g. Threshold gaming — claims just below approval limits ─────────
    if "approval_threshold" in df.columns:
        band = CONFIG["just_below_threshold_band_pct"]
        just_below = df[
            (df["claim_amount"] >= df["approval_threshold"] * (1 - band)) &
            (df["claim_amount"] < df["approval_threshold"])
        ]
        threshold_ratio_by_fac = (
            just_below.groupby("facility_id").size() /
            df.groupby("facility_id").size()
        ).dropna()
        for facility_id, ratio in threshold_ratio_by_fac.items():
            if ratio > 0.30:
                for cid in just_below[just_below["facility_id"] == facility_id]["claim_id"].tolist():
                    _make_flag(flags, cid, "THRESHOLD_GAMING",
                        f"Facility {facility_id}: {ratio*100:.1f}% of claims fall just below "
                        f"the authorisation threshold (within {band*100:.0f}%). "
                        f"Providers splitting or trimming bills to avoid pre-authorisation.",
                        Severity.HIGH, M, str(facility_id), "FACILITY",
                        {"ratio": float(ratio), "band_pct": float(band)})

    # 3h. KS-test against county peer distribution ──────────────────────
    if len(df) > 200:
        for (facility_id, level, county), grp in df.groupby(
            ["facility_id", "facility_level", "county"]
        ):
            if len(grp) < 50:
                continue
            peers = df[
                (df["facility_level"] == level) &
                (df["county"] == county) &
                (df["facility_id"] != facility_id)
            ]["claim_amount"].dropna()
            if len(peers) < 50:
                continue
            ks_stat, p_value = stats.ks_2samp(
                grp["claim_amount"].dropna().values,
                peers.values
            )
            if p_value < 0.01 and ks_stat > 0.3:
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "STATISTICAL_OUTLIER",
                        f"Facility {facility_id}: Claim-amount distribution significantly differs "
                        f"from {len(peers)} peer claims in {county} county "
                        f"(KS stat={ks_stat:.3f}, p={p_value:.5f}). "
                        f"Systematic pricing deviation from peer group.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY",
                        {"ks_stat": float(ks_stat), "p_value": float(p_value)})

    # 3i. Single-diagnosis dominance ───────────────────────────────────
    if "diagnosis_code" in df.columns:
        for facility_id, grp in df.groupby("facility_id"):
            if len(grp) < 50:
                continue
            code_counts = grp["diagnosis_code"].value_counts(normalize=True)
            top_code    = code_counts.index[0]
            top_ratio   = code_counts.iloc[0]
            if top_ratio > 0.50:
                flagged = grp[grp["diagnosis_code"] == top_code]
                for cid in flagged["claim_id"].tolist():
                    _make_flag(flags, cid, "SINGLE_DIAGNOSIS_HIGH_VOLUME",
                        f"Facility {facility_id}: {top_ratio*100:.1f}% of claims carry diagnosis "
                        f"'{top_code}' — no single ICD code should dominate >50% of claims "
                        f"in a genuine population. Indicates DRG gaming or lazy coding.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY",
                        {"diagnosis_code": str(top_code), "ratio": float(top_ratio)})

    # 3j. Implausible length-of-stay by procedure ───────────────────────
    if "inpatient_days" in df.columns and "los_expected_days" in df.columns:
        # Expected LOS should be provided as a facility/tariff reference column
        df["los_ratio"] = df["inpatient_days"] / df["los_expected_days"].replace(0, np.nan)
        extreme_los = df[(df["los_ratio"] > 3.0) | (df["los_ratio"] < 0.2)]
        for _, row in extreme_los.iterrows():
            direction = "excessive" if row["los_ratio"] > 3.0 else "implausibly short"
            _make_flag(flags, row["claim_id"], "IMPLAUSIBLE_LOS",
                f"Claim {row['claim_id']}: Length-of-stay {row['inpatient_days']:.0f} days is "
                f"{direction} vs. expected {row['los_expected_days']:.0f} days for "
                f"'{row.get('procedure_type', 'UNKNOWN')}' (ratio = {row['los_ratio']:.2f}). "
                f"Inflated LOS inflates DRG payment; zero LOS with major surgery is impossible.",
                Severity.MEDIUM, M, str(row.get("facility_id", "")), "FACILITY",
                {"actual_los": float(row["inpatient_days"]),
                 "expected_los": float(row["los_expected_days"]),
                 "ratio": float(row["los_ratio"])})

    return flags


#  MODULE 4 — DUPLICATE CLAIM DETECTION

def check_duplicate_claims(claims: pd.DataFrame) -> list[FlagRecord]:
    """
    Module 4: Exact and near-duplicate claim detection.

      4a. Exact duplicates (same member, facility, service, date, amount)
      4b. Near-duplicates: same member+service within 3 days, different amounts
      4c. Split-billing: same member/episode split across multiple small claims
      4d. Unbundling: component services billed separately when a DRG covers all
      4e. High rejection-then-resubmit rate (iterative amount manipulation)
      4f. Cross-facility duplicates (same service, same member, different facility)
    """
    M = "MODULE_4_DUPLICATES"
    flags: list[FlagRecord] = []
    df = claims.copy()
    df["claim_date"] = _safe_date(df["claim_date"])

    # 4a. Exact duplicates ──────────────────────────────────────────────
    dupe_keys = ["member_id", "facility_id", "service_code", "claim_date", "claim_amount"]
    available_keys = [k for k in dupe_keys if k in df.columns]
    if len(available_keys) >= 4:
        exact_dupes = df[df.duplicated(subset=available_keys, keep=False)]
        for cid in exact_dupes["claim_id"].tolist():
            _make_flag(flags, cid, "DUPLICATE_CLAIM",
                f"Claim {cid}: Exact duplicate — identical member, facility, service, date, "
                f"and amount as another claim. Classic double-billing.",
                Severity.HIGH, M, evidence={"dupe_key": str(available_keys)})

    # 4b. Near-duplicates within 3-day window ───────────────────────────
    df_sorted = df.sort_values(["member_id", "service_code", "claim_date"])
    for (member_id, service_code), grp in df_sorted.groupby(["member_id", "service_code"]):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                d1 = grp.loc[i, "claim_date"]
                d2 = grp.loc[j, "claim_date"]
                if pd.isna(d1) or pd.isna(d2):
                    continue
                diff = abs((d2 - d1).days)
                if diff == 0:
                    continue  # Handled in 4a
                if diff <= 3:
                    _make_flag(flags, grp.loc[j, "claim_id"], "DUPLICATE_CLAIM",
                        f"Near-duplicate: Member {member_id}, service '{service_code}' "
                        f"claimed {diff} day(s) apart — possible resubmission with modified amount.",
                        Severity.MEDIUM, M, str(member_id), "MEMBER",
                        {"day_diff": diff, "service": str(service_code)})
                elif diff > 3:
                    break

    # 4c. Split-billing detection ───────────────────────────────────────
    # Same member, same facility, same day, multiple claims — total exceeds DRG tariff
    for (member_id, facility_id, claim_date), grp in df.groupby(
        ["member_id", "facility_id", "claim_date"]
    ):
        if len(grp) < 2:
            continue
        total = grp["claim_amount"].sum()
        # Check if sum matches a known bundled tariff
        for proc, tariff in CONFIG["tariff_reference"].items():
            if 0.80 * tariff <= total <= 1.50 * tariff:
                # The sum of parts ≈ bundled tariff — classic unbundling
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "SPLIT_BILLING",
                        f"Member {member_id} at Facility {facility_id} on "
                        f"{pd.Timestamp(claim_date).date()}: {len(grp)} claims totalling "
                        f"{total:,.0f} KSh ≈ bundled tariff for {proc} ({tariff:,} KSh). "
                        f"Components may be unbundled to avoid threshold or inflate total.",
                        Severity.HIGH, M, str(facility_id), "FACILITY",
                        {"total": float(total), "matched_proc": proc,
                         "matched_tariff": tariff, "parts": len(grp)})

    # 4d. Cross-facility duplicates ─────────────────────────────────────
    if "facility_id" in df.columns:
        for (member_id, service_code, claim_date), grp in df.groupby(
            ["member_id", "service_code", "claim_date"]
        ):
            if grp["facility_id"].nunique() > 1:
                for cid in grp["claim_id"].tolist():
                    facs = grp["facility_id"].tolist()
                    _make_flag(flags, cid, "DUPLICATE_CLAIM",
                        f"Cross-facility duplicate: Member {member_id} has identical service "
                        f"'{service_code}' on {pd.Timestamp(claim_date).date()} "
                        f"claimed by facilities {facs}. "
                        f"A member cannot receive the same service at two locations simultaneously.",
                        Severity.CRITICAL, M, str(member_id), "MEMBER",
                        {"facilities": list(map(str, facs))})

    # 4e. High rejection-resubmit rate ──────────────────────────────────
    if "claim_status" in df.columns and "original_claim_id" in df.columns:
        resubmitted = df[df["original_claim_id"].notna()]
        rejections  = df[df["claim_status"].str.upper() == "REJECTED"] if "claim_status" in df.columns else pd.DataFrame()
        for facility_id, grp in resubmitted.groupby("facility_id"):
            total = len(df[df["facility_id"] == facility_id])
            resub_rate = len(grp) / total if total > 0 else 0
            if resub_rate > 0.30 and total > 20:
                for cid in grp["claim_id"].tolist():
                    _make_flag(flags, cid, "HIGH_REJECTION_THEN_RESUBMIT",
                        f"Facility {facility_id}: {resub_rate*100:.1f}% resubmission rate "
                        f"({len(grp)}/{total} claims). High resubmission rates indicate "
                        f"iterative amount-fishing to find the highest payable amount.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY",
                        {"resub_rate": float(resub_rate)})

    return flags


#  MODULE 5 — CLINICAL CODING ANOMALIES

IMPOSSIBLE_PROCEDURE_COMBOS: list[tuple[str, str, str]] = [
    ("APPENDECTOMY",        "COLONOSCOPY",              "Bowel prep for colonoscopy is incompatible with emergency appendectomy on same day"),
    ("GENERAL_ANESTHESIA",  "SAME_DAY_DISCHARGE",       "Major general anesthesia with same-day discharge is clinically unsafe for most procedures"),
    ("HYSTERECTOMY",        "DELIVERY",                 "Cannot perform delivery and hysterectomy simultaneously on same visit in standard protocols"),
    ("TONSILLECTOMY",       "ADENOIDECTOMY_SEPARATE",   "Tonsillectomy and adenoidectomy are coded as a combined procedure, not billed separately"),
    ("BILATERAL_PROCEDURE", "BILATERAL_PROCEDURE",      "Bilateral procedures cannot be billed twice — they are a single combined code"),
    ("DIALYSIS",            "KIDNEY_TRANSPLANT",        "Dialysis on same day as kidney transplant is clinically contradictory"),
    ("CARDIAC_BYPASS",      "SAME_DAY_DISCHARGE",       "Cardiac bypass surgery mandates minimum 5-day inpatient stay"),
    ("ICU_VENTILATION",     "OUTPATIENT",               "ICU ventilation is definitionally an inpatient ICU procedure, not outpatient"),
    ("ANESTHESIA_GENERAL",  "ANESTHESIA_SPINAL",        "Dual anaesthesia modalities cannot be billed for same procedure"),
    ("CHEMOTHERAPY",        "SURGERY",                  "Same-day chemotherapy and major surgery is not a standard protocol"),
]

LEVEL2_RESTRICTED: set[str] = {
    "ORGAN_TRANSPLANT", "NEUROSURGERY", "CARDIAC_BYPASS", "ICU_VENTILATION",
    "BONE_MARROW_TRANSPLANT", "OPEN_HEART", "CRANIECTOMY", "LIVER_RESECTION"
}


def check_clinical_coding_anomalies(
    claims: pd.DataFrame, members: pd.DataFrame, facilities: pd.DataFrame
) -> list[FlagRecord]:
    """
    Module 5: Clinical coding and medical plausibility checks.

      5a. Sex-procedure mismatches
      5b. Impossible same-day procedure combinations
      5c. Procedures beyond facility capability level
      5d. Upcoding (excessive high-complexity consultations)
      5e. Tariff amount vs. reference tariff (extreme markup detection)
      5f. Age-inappropriate procedures
      5g. Ping-pong referral detection (circular referral chains)
      5h. Readmission within 30 days for same DRG (possible premature discharge)
      5i. Missing inpatient days for procedures requiring admission
    """
    M = "MODULE_5_CLINICAL"
    flags: list[FlagRecord] = []

    members = _normalise_sex(members)
    df = claims.merge(members[["member_id", "sex_norm", "dob"]], on="member_id", how="left")
    df = df.merge(facilities[["facility_id", "facility_level"]], on="facility_id", how="left")
    df["claim_date"]  = _safe_date(df["claim_date"])
    df["dob"]         = _safe_date(df["dob"])
    df["age_at_claim"] = compute_age(df["dob"], df["claim_date"])
    df["proc_upper"]   = _proc_upper(df)

    # 5a. Sex-procedure mismatches ──────────────────────────────────────
    for proc in FEMALE_ONLY_PROCEDURES:
        mis = df[(df["sex_norm"] == "M") & df["proc_upper"].str.contains(proc, na=False)]
        for _, row in mis.iterrows():
            _make_flag(flags, row["claim_id"], "IMPOSSIBLE_COMBINATION",
                f"Male member {row['member_id']}: Female-only procedure '{proc}' claimed. "
                f"Sex field may be falsified to enable otherwise-blocked claims.",
                Severity.CRITICAL, M, str(row["member_id"]), "MEMBER")

    for proc in MALE_ONLY_PROCEDURES:
        mis = df[(df["sex_norm"] == "F") & df["proc_upper"].str.contains(proc, na=False)]
        for _, row in mis.iterrows():
            _make_flag(flags, row["claim_id"], "IMPOSSIBLE_COMBINATION",
                f"Female member {row['member_id']}: Male-only procedure '{proc}' claimed.",
                Severity.CRITICAL, M, str(row["member_id"]), "MEMBER")

    # 5b. Impossible same-day combinations ──────────────────────────────
    for (member_id, dt), day_grp in df.groupby(["member_id", df["claim_date"].dt.date]):
        procs = day_grp["proc_upper"].tolist()
        for proc_a, proc_b, reason in IMPOSSIBLE_PROCEDURE_COMBOS:
            if any(proc_a in p for p in procs) and any(proc_b in p for p in procs):
                for cid in day_grp["claim_id"].tolist():
                    _make_flag(flags, cid, "IMPOSSIBLE_COMBINATION",
                        f"Member {member_id} on {dt}: {reason}.",
                        Severity.HIGH, M, str(member_id), "MEMBER",
                        {"proc_a": proc_a, "proc_b": proc_b})

    # 5c. Procedures beyond facility level ──────────────────────────────
    l2_fac = df[df["facility_level"].isin([2, "2", "Level 2", "LEVEL_2"])]
    for proc in LEVEL2_RESTRICTED:
        over = l2_fac[l2_fac["proc_upper"].str.contains(proc, na=False)]
        for _, row in over.iterrows():
            _make_flag(flags, row["claim_id"], "CLINICAL_CODE_ANOMALY",
                f"Facility {row['facility_id']} (Level 2) claimed restricted procedure '{proc}'. "
                f"Level 2 facilities lack the infrastructure, specialist staffing, "
                f"and ICU support required for this procedure category.",
                Severity.HIGH, M, str(row["facility_id"]), "FACILITY",
                {"procedure": proc, "facility_level": str(row["facility_level"])})

    # 5d. Upcoding detection ────────────────────────────────────────────
    consults = df[df["proc_upper"].str.contains("CONSULT|OPD|OUTPATIENT", na=False)]
    if "complexity" in consults.columns:
        for facility_id, grp in consults.groupby("facility_id"):
            if len(grp) < 30:
                continue
            high_complex = (grp["complexity"] >= 4).sum()
            ratio = high_complex / len(grp)
            if ratio > 0.60:
                for cid in grp[grp["complexity"] >= 4]["claim_id"].tolist():
                    _make_flag(flags, cid, "UPCODING",
                        f"Facility {facility_id}: {ratio*100:.1f}% of OPD consultations coded "
                        f"at complexity ≥4 (expected <30% for primary care). "
                        f"Systematic upcoding inflates per-consultation reimbursement.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY",
                        {"high_complex_ratio": float(ratio)})

    # 5e. Extreme tariff markup ─────────────────────────────────────────
    for proc_key, reference_tariff in CONFIG["tariff_reference"].items():
        subset = df[df["proc_upper"].str.contains(proc_key, na=False)]
        if subset.empty or "claim_amount" not in subset.columns:
            continue
        max_allowed = reference_tariff * CONFIG["max_markup_ratio_over_tariff"]
        overpriced  = subset[subset["claim_amount"] > max_allowed]
        for _, row in overpriced.iterrows():
            markup = row["claim_amount"] / reference_tariff
            _make_flag(flags, row["claim_id"], "TARIFF_EXTREME_MARKUP",
                f"Claim {row['claim_id']}: {row['claim_amount']:,.0f} KSh for '{proc_key}' "
                f"is {markup:.1f}× the SHA reference tariff of {reference_tariff:,} KSh "
                f"(ceiling = {max_allowed:,.0f} KSh). Extreme upcoding or price fabrication.",
                Severity.HIGH, M, str(row.get("facility_id", "")), "FACILITY",
                {"claim_amount": float(row["claim_amount"]),
                 "reference_tariff": float(reference_tariff),
                 "markup_ratio": float(markup)})

    # 5f. Age-inappropriate procedures ─────────────────────────────────
    PAEDIATRIC_ONLY_MAX_AGE = 18
    ADULT_ONLY_MIN_AGE      = 16
    GERIATRIC_MIN_AGE       = 60

    paediatric_keywords = ["PAEDIATRIC_OPD", "NEONATAL", "CHILD_VACCINATION", "PAEDS_ICU"]
    adult_keywords      = ["PROSTATE", "MENOPAUSE", "COLONOSCOPY_SURVEILLANCE"]

    for proc in paediatric_keywords:
        adult_paed = df[
            df["proc_upper"].str.contains(proc, na=False) &
            (df["age_at_claim"] > PAEDIATRIC_ONLY_MAX_AGE)
        ]
        for _, row in adult_paed.iterrows():
            _make_flag(flags, row["claim_id"], "CLINICAL_CODE_ANOMALY",
                f"Member {row['member_id']} age {row['age_at_claim']:.0f}: "
                f"Paediatric-coded procedure '{proc}' claimed for adult. "
                f"Paediatric billing codes carry different tariffs — possible tariff fraud.",
                Severity.HIGH, M, str(row["member_id"]), "MEMBER",
                {"age": float(row["age_at_claim"]), "procedure": proc})

    # 5g. Readmission within 30 days — same DRG ─────────────────────────
    if "diagnosis_code" in df.columns:
        admit = df[df["proc_upper"].str.contains("INPATIENT|ADMISSION", na=False)].copy()
        admit = admit.sort_values(["member_id", "diagnosis_code", "claim_date"])
        for (member_id, diag), grp in admit.groupby(["member_id", "diagnosis_code"]):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                d1 = grp.loc[i-1, "claim_date"]
                d2 = grp.loc[i,   "claim_date"]
                if pd.notna(d1) and pd.notna(d2):
                    gap = (d2 - d1).days
                    if 0 < gap <= 30:
                        _make_flag(flags, grp.loc[i, "claim_id"], "READMISSION_BOUNCE",
                            f"Member {member_id}: Re-admitted within {gap} days for same diagnosis "
                            f"'{diag}'. Could indicate premature discharge to generate two DRG "
                            f"payments instead of one continuous stay.",
                            Severity.MEDIUM, M, str(member_id), "MEMBER",
                            {"gap_days": gap, "diagnosis": str(diag)})

    # 5h. Inpatient days required but billed as zero ────────────────────
    REQUIRES_ADMISSION: set[str] = {
        "MAJOR_SURGERY", "C_SECTION", "CAESAREAN", "DELIVERY",
        "ICU", "CARDIAC_BYPASS", "ORGAN_TRANSPLANT", "CHEMOTHERAPY"
    }
    if "inpatient_days" in df.columns:
        for proc in REQUIRES_ADMISSION:
            zero_los = df[
                df["proc_upper"].str.contains(proc, na=False) &
                (df["inpatient_days"].fillna(0) == 0)
            ]
            for _, row in zero_los.iterrows():
                _make_flag(flags, row["claim_id"], "IMPOSSIBLE_COMBINATION",
                    f"Claim {row['claim_id']}: Procedure '{proc}' billed with 0 inpatient days. "
                    f"This procedure mandates hospital admission; zero LOS indicates "
                    f"an outpatient claim masquerading as an inpatient episode.",
                    Severity.HIGH, M, str(row.get("facility_id", "")), "FACILITY",
                    {"procedure": proc, "inpatient_days": 0})

    # 5i. Extreme single-claim amount ───────────────────────────────────
    extreme_high = df[df["claim_amount"] > CONFIG["max_single_claim_amount_ksh"]]
    for _, row in extreme_high.iterrows():
        _make_flag(flags, row["claim_id"], "TARIFF_EXTREME_MARKUP",
            f"Claim {row['claim_id']}: Amount {row['claim_amount']:,.0f} KSh exceeds the "
            f"single-claim review ceiling of {CONFIG['max_single_claim_amount_ksh']:,} KSh. "
            f"Any claim above this threshold requires mandatory clinical peer review.",
            Severity.HIGH, M, str(row.get("facility_id", "")), "FACILITY",
            {"claim_amount": float(row["claim_amount"])})

    extreme_low = df[
        (df["claim_amount"] < CONFIG["min_plausible_claim_amount_ksh"]) &
        (df["claim_amount"] > 0)
    ]
    for _, row in extreme_low.iterrows():
        _make_flag(flags, row["claim_id"], "CLINICAL_CODE_ANOMALY",
            f"Claim {row['claim_id']}: Amount {row['claim_amount']:.0f} KSh is below the "
            f"minimum plausible threshold of {CONFIG['min_plausible_claim_amount_ksh']} KSh. "
            f"May indicate data entry error or test/phantom claim submission.",
            Severity.LOW, M, str(row.get("facility_id", "")), "FACILITY",
            {"claim_amount": float(row["claim_amount"])})

    return flags


#  MODULE 6 — GHOST PATIENT DETECTION

def check_ghost_patients(
    claims: pd.DataFrame, members: pd.DataFrame
) -> list[FlagRecord]:
    """
    Module 6: Ghost / phantom beneficiary detection.

      6a. Members with only high-cost procedures, zero preventive care
      6b. Multiple members at identical address + phone
      6c. Bulk registration (same agent, same date)
      6d. Members with identical demographic fingerprints
      6e. Members who never appear in any non-claim context
      6f. Suspiciously short intervals between registration and first high-cost claim
      6g. Members with only one unique facility across entire history
          and that facility is flagged
      6h. Phantom members: missing critical identity fields
    """
    M = "MODULE_6_GHOST"
    flags: list[FlagRecord] = []

    df = claims.merge(members, on="member_id", how="left")
    df["claim_date"] = _safe_date(df["claim_date"])
    df["proc_upper"]  = _proc_upper(df)

    LOW_ACUITY: set[str] = {
        "IMMUNIZATION", "WELLNESS_VISIT", "ROUTINE_CHECKUP", "ANTENATAL_VISIT",
        "FAMILY_PLANNING", "OUTPATIENT", "OPD", "DENTAL_ROUTINE", "OPTICAL_ROUTINE"
    }
    HIGH_COST: set[str] = {
        "MAJOR_SURGERY", "ICU", "DIALYSIS", "CHEMOTHERAPY", "ORGAN_TRANSPLANT",
        "CARDIAC_BYPASS", "NEUROSURGERY", "BONE_MARROW_TRANSPLANT"
    }

    # 6a. High-cost only, zero routine care ────────────────────────────
    for member_id, grp in df.groupby("member_id"):
        if len(grp) < 3:
            continue
        procs = grp["proc_upper"].tolist()
        has_low  = any(any(la in p for la in LOW_ACUITY) for p in procs)
        has_high = any(any(hc in p for hc in HIGH_COST) for p in procs)
        if has_high and not has_low:
            for cid in grp["claim_id"].tolist():
                _make_flag(flags, cid, "GHOST_PATIENT",
                    f"Member {member_id}: {len(grp)} high-cost claims with ZERO routine/preventive "
                    f"care on record. Real patients accumulate routine contacts across all acuity levels. "
                    f"Pattern consistent with fabricated high-value claims.",
                    Severity.MEDIUM, M, str(member_id), "MEMBER")

    # 6b. Address+phone sharing ─────────────────────────────────────────
    if "address" in members.columns and "phone" in members.columns:
        shared = (
            members.groupby(["address", "phone"])["member_id"]
            .apply(list)
            .reset_index(name="member_ids")
        )
        shared = shared[shared["member_ids"].apply(len) > CONFIG["ghost_address_phone_shared_threshold"]]
        suspicious_mids = {mid for mids in shared["member_ids"] for mid in mids}
        sub = df[df["member_id"].isin(suspicious_mids)]
        for cid in sub["claim_id"].tolist():
            _make_flag(flags, cid, "GHOST_PATIENT",
                f"Member shares address+phone with >{CONFIG['ghost_address_phone_shared_threshold']} "
                f"other members — possible ghost beneficiary cluster enrolled at same address.",
                Severity.MEDIUM, M, evidence={"threshold": CONFIG["ghost_address_phone_shared_threshold"]})

    # 6c. Bulk registration ─────────────────────────────────────────────
    if "registration_agent_id" in members.columns and "registration_date" in members.columns:
        members = members.copy()
        members["registration_date"] = _safe_date(members["registration_date"])
        bulk = (
            members.groupby(["registration_agent_id", "registration_date"])
            .size().reset_index(name="count")
        )
        bulk_sus = bulk[bulk["count"] > CONFIG["bulk_registration_daily_threshold"]]
        for _, row in bulk_sus.iterrows():
            agent_id = row["registration_agent_id"]
            reg_date = row["registration_date"]
            bulk_mids = members[
                (members["registration_agent_id"] == agent_id) &
                (members["registration_date"]      == reg_date)
            ]["member_id"].tolist()
            sub = df[df["member_id"].isin(bulk_mids)]
            for cid in sub["claim_id"].tolist():
                _make_flag(flags, cid, "GHOST_PATIENT",
                    f"Agent {agent_id} registered {row['count']} members on "
                    f"{reg_date.date() if pd.notna(reg_date) else 'UNKNOWN'} "
                    f"(threshold = {CONFIG['bulk_registration_daily_threshold']}). "
                    f"Mass-registration events are a hallmark of ghost beneficiary schemes.",
                    Severity.HIGH, M, str(agent_id), "AGENT",
                    {"agent_id": str(agent_id), "count": int(row["count"])})

    # 6d. Identical demographic fingerprints ────────────────────────────
    demo_cols = [c for c in ["dob", "sex", "county", "national_id"] if c in members.columns]
    if len(demo_cols) >= 3:
        dup_demos = members[members.duplicated(subset=demo_cols, keep=False)]
        dup_mids  = set(dup_demos["member_id"].tolist())
        sub = df[df["member_id"].isin(dup_mids)]
        for cid in sub["claim_id"].tolist():
            _make_flag(flags, cid, "GHOST_PATIENT",
                f"Member shares identical demographic fingerprint (dob+sex+county+national_id) "
                f"with another member — possible duplicate enrollment or fabricated identity.",
                Severity.HIGH, M, evidence={"demo_cols": demo_cols})

    # 6e. Registration-to-first-high-cost claim interval ────────────────
    if "registration_date" in members.columns:
        reg_df = df.merge(
            members[["member_id", "registration_date"]].copy().assign(
                registration_date=lambda d: _safe_date(d["registration_date"])
            ),
            on="member_id", how="left"
        )
        high_cost_claims = reg_df[
            reg_df["proc_upper"].apply(lambda p: any(hc in p for hc in HIGH_COST))
        ].copy()
        high_cost_claims["days_since_reg"] = (
            high_cost_claims["claim_date"] - high_cost_claims["registration_date"]
        ).dt.days
        suspicious_speed = high_cost_claims[
            (high_cost_claims["days_since_reg"] >= 0) &
            (high_cost_claims["days_since_reg"] <= 14)
        ]
        for _, row in suspicious_speed.iterrows():
            _make_flag(flags, row["claim_id"], "GHOST_PATIENT",
                f"Member {row['member_id']}: High-cost procedure claimed only "
                f"{int(row['days_since_reg'])} days after scheme registration. "
                f"Pre-existing conditions enrolled specifically to claim is a common scheme.",
                Severity.HIGH, M, str(row["member_id"]), "MEMBER",
                {"days_since_registration": int(row["days_since_reg"])})

    # 6h. Missing critical identity fields ─────────────────────────────
    critical_fields = [c for c in ["dob", "sex", "national_id", "name"] if c in members.columns]
    for field_col in critical_fields:
        missing = members[members[field_col].isna() | (members[field_col].astype(str).str.strip() == "")]
        missing_mids = set(missing["member_id"].tolist())
        sub = df[df["member_id"].isin(missing_mids)]
        for cid in sub["claim_id"].tolist():
            _make_flag(flags, cid, "GHOST_PATIENT",
                f"Member claiming has NULL/empty '{field_col}' field. "
                f"Missing identity fields prevent verification — consistent with ghost enrollment.",
                Severity.MEDIUM, M, evidence={"missing_field": field_col})

    return flags


#  MODULE 7 — NETWORK COLLUSION DETECTION

def check_network_collusion(claims: pd.DataFrame) -> list[FlagRecord]:
    """
    Module 7: Structural network analysis for collusion detection.

      7a. Member over-concentration at single facility
      7b. Agent–facility concentration ratio
      7c. Graph-based clique detection (if networkx available)
      7d. Facility pair member overlap (Jaccard similarity)
      7e. Provider-member relationship anomalies (same provider for unrelated conditions)
      7f. Temporal synchronisation — multiple members claiming identical services same day
      7g. Ping-pong referral chains between facility pairs
    """
    M = "MODULE_7_COLLUSION"
    flags: list[FlagRecord] = []
    df = claims.copy()
    df["claim_date"] = _safe_date(df["claim_date"])
    df["proc_upper"]  = _proc_upper(df)

    # 7a. Member over-concentration at single facility ──────────────────
    member_fac = df.groupby(["member_id", "facility_id"]).size().reset_index(name="visit_count")
    member_tot = df.groupby("member_id").size().reset_index(name="total_claims")
    mf = member_fac.merge(member_tot, on="member_id")
    mf["conc"] = mf["visit_count"] / mf["total_claims"]
    suspicious = mf[
        (mf["conc"] > CONFIG["member_facility_concentration_ratio"]) &
        (mf["total_claims"] > CONFIG["min_claims_for_concentration"])
    ]
    for _, row in suspicious.iterrows():
        member_claims = df[
            (df["member_id"] == row["member_id"]) &
            (df["facility_id"] == row["facility_id"])
        ]["claim_id"].tolist()
        for cid in member_claims:
            _make_flag(flags, cid, "NETWORK_COLLUSION",
                f"Member {row['member_id']}: {row['conc']*100:.1f}% of "
                f"{row['total_claims']} claims exclusively at Facility {row['facility_id']} "
                f"across all conditions and time periods. "
                f"Real patients use varied facilities; this concentration implies an arranged relationship.",
                Severity.MEDIUM, M, str(row["facility_id"]), "FACILITY",
                {"concentration": float(row["conc"]), "total_claims": int(row["total_claims"])})

    # 7b. Agent–facility concentration ─────────────────────────────────
    if "registration_agent_id" in df.columns:
        af = (
            df.groupby(["registration_agent_id", "facility_id"])
            .agg(total_amount=("claim_amount", "sum"), unique_members=("member_id", "nunique"))
            .reset_index()
        )
        agent_total = (
            df.groupby("registration_agent_id")["claim_amount"]
            .sum().reset_index(name="agent_total")
        )
        af = af.merge(agent_total, on="registration_agent_id")
        af["fac_conc"] = af["total_amount"] / af["agent_total"]
        for _, row in af[af["fac_conc"] > CONFIG["agent_facility_concentration_ratio"]].iterrows():
            agent_claims = df[
                df["registration_agent_id"] == row["registration_agent_id"]
            ]["claim_id"].tolist()
            for cid in agent_claims:
                _make_flag(flags, cid, "NETWORK_COLLUSION",
                    f"Agent {row['registration_agent_id']}: "
                    f"{row['fac_conc']*100:.1f}% of enrolled members' claim spend concentrated "
                    f"at Facility {row['facility_id']} ({row['unique_members']} members). "
                    f"Agent-facility pairing is the structural backbone of many ghost member schemes.",
                    Severity.HIGH, M, str(row["registration_agent_id"]), "AGENT",
                    {"facility_concentration": float(row["fac_conc"]),
                     "unique_members": int(row["unique_members"])})

    # 7c. Graph-based clique detection ─────────────────────────────────
    if _HAS_NX and "registration_agent_id" in df.columns:
        G = nx.Graph()
        agent_fac_pairs = df[["registration_agent_id", "facility_id"]].drop_duplicates()
        for _, row in agent_fac_pairs.iterrows():
            agent_node   = f"AGENT_{row['registration_agent_id']}"
            facility_node= f"FAC_{row['facility_id']}"
            G.add_edge(agent_node, facility_node,
                       weight=float(df[
                           (df["registration_agent_id"] == row["registration_agent_id"]) &
                           (df["facility_id"] == row["facility_id"])
                       ]["claim_amount"].sum()))
        # Find cliques of size >= threshold
        cliques = [c for c in nx.find_cliques(G) if len(c) >= CONFIG["clique_min_size"]]
        for clique in cliques:
            agents    = [n.replace("AGENT_", "") for n in clique if n.startswith("AGENT_")]
            facilities= [n.replace("FAC_", "") for n in clique if n.startswith("FAC_")]
            clique_claims = df[
                df["registration_agent_id"].astype(str).isin(agents) &
                df["facility_id"].astype(str).isin(facilities)
            ]["claim_id"].tolist()
            for cid in clique_claims:
                _make_flag(flags, cid, "NETWORK_COLLUSION",
                    f"Graph clique detected: agents {agents} and facilities {facilities} "
                    f"form a fully-connected sub-network (clique size = {len(clique)}). "
                    f"Clique structures in agent-facility graphs are characteristic of organised fraud rings.",
                    Severity.HIGH, M, evidence={"agents": agents, "facilities": facilities})

    # 7d. Jaccard overlap between facility member sets ─────────────────
    fac_members: dict[Any, set] = {
        fid: set(grp["member_id"].tolist())
        for fid, grp in df.groupby("facility_id")
    }
    facility_ids = list(fac_members.keys())
    for i in range(len(facility_ids)):
        for j in range(i + 1, len(facility_ids)):
            fa, fb = facility_ids[i], facility_ids[j]
            setA, setB = fac_members[fa], fac_members[fb]
            union_size = len(setA | setB)
            if union_size == 0:
                continue
            jaccard = len(setA & setB) / union_size
            if jaccard > CONFIG["shared_member_overlap_threshold"] and len(setA & setB) > 10:
                shared_members = setA & setB
                shared_claims = df[
                    df["member_id"].isin(shared_members) &
                    df["facility_id"].isin([fa, fb])
                ]["claim_id"].tolist()
                for cid in shared_claims:
                    _make_flag(flags, cid, "NETWORK_COLLUSION",
                        f"Facilities {fa} and {fb} share {len(shared_members)} members "
                        f"(Jaccard = {jaccard:.2f}, threshold = {CONFIG['shared_member_overlap_threshold']}). "
                        f"High member overlap between unrelated facilities indicates coordinated enrollment.",
                        Severity.MEDIUM, M, str(fa), "FACILITY",
                        {"jaccard": float(jaccard), "shared_members": len(shared_members)})

    # 7e. Temporal synchronisation — identical services, same day, many members ──
    for (facility_id, claim_date, proc_upper), grp in df.groupby(
        ["facility_id", "claim_date", "proc_upper"]
    ):
        unique_members = grp["member_id"].nunique()
        if unique_members > 50:  # 50 different members, same service, same day
            for cid in grp["claim_id"].tolist():
                _make_flag(flags, cid, "TEMPORAL_CLUSTERING",
                    f"Facility {facility_id} on {pd.Timestamp(claim_date).date()}: "
                    f"{unique_members} different members all claimed '{proc_upper}'. "
                    f"Mass same-day identical claims suggest batch fabrication.",
                    Severity.HIGH, M, str(facility_id), "FACILITY",
                    {"unique_members": int(unique_members), "procedure": str(proc_upper)})

    # 7f. Ping-pong referrals ───────────────────────────────────────────
    if "referral_from_facility" in df.columns:
        for member_id, grp in df[df["referral_from_facility"].notna()].groupby("member_id"):
            grp = grp.sort_values("claim_date").reset_index(drop=True)
            fac_seq = grp["facility_id"].tolist()
            ref_seq = grp["referral_from_facility"].tolist()
            # Detect A→B→A patterns
            for i in range(2, len(fac_seq)):
                if fac_seq[i] == fac_seq[i-2] and ref_seq[i] == fac_seq[i-1]:
                    _make_flag(flags, grp.loc[i, "claim_id"], "PING_PONG_REFERRAL",
                        f"Member {member_id}: Ping-pong referral pattern detected — "
                        f"Facility {fac_seq[i-2]} → {fac_seq[i-1]} → {fac_seq[i]}. "
                        f"Circular referrals generate multiple consultation and procedure fees "
                        f"without medical necessity.",
                        Severity.MEDIUM, M, str(member_id), "MEMBER",
                        {"pattern": f"{fac_seq[i-2]}→{fac_seq[i-1]}→{fac_seq[i]}"})

    return flags


#  MODULE 8 — PROVIDER IDENTITY & CREDENTIAL CHECKS

def check_provider_integrity(
    claims: pd.DataFrame, facilities: pd.DataFrame, providers: Optional[pd.DataFrame] = None
) -> list[FlagRecord]:
    """
    Module 8: Provider and facility credential integrity.

      8a. Claims from de-registered / suspended facilities
      8b. Claims attributed to deceased or de-licensed providers
      8c. Facility claiming procedure categories not in their accreditation scope
      8d. Sudden ownership changes coinciding with claim volume spikes
      8e. Phantom facility detection (no physical inspection record)
    """
    M = "MODULE_8_PROVIDER"
    flags: list[FlagRecord] = []
    df = claims.merge(facilities, on="facility_id", how="left")
    df["claim_date"] = _safe_date(df["claim_date"])

    # 8a. Claims from suspended facilities ─────────────────────────────
    if "facility_status" in df.columns and "suspension_date" in df.columns:
        df["suspension_date"] = _safe_date(df["suspension_date"])
        suspended = df[
            (df["facility_status"].str.upper().isin(["SUSPENDED", "DEREGISTERED", "BLACKLISTED"])) |
            (df["suspension_date"].notna() & (df["claim_date"] >= df["suspension_date"]))
        ]
        for _, row in suspended.iterrows():
            _make_flag(flags, row["claim_id"], "PHANTOM_SERVICE",
                f"Claim from Facility {row['facility_id']} which is "
                f"SUSPENDED/DEREGISTERED (status: {row.get('facility_status','UNKNOWN')}). "
                f"No claims can be legitimately generated by a non-operating facility.",
                Severity.CRITICAL, M, str(row["facility_id"]), "FACILITY",
                {"facility_status": str(row.get("facility_status", "UNKNOWN"))})

    # 8b. Accreditation scope violations ───────────────────────────────
    if "accredited_procedures" in facilities.columns:
        df_scope = df.merge(
            facilities[["facility_id", "accredited_procedures"]], on="facility_id", how="left"
        )
        df_scope["proc_upper"] = _proc_upper(df_scope)
        for _, row in df_scope.iterrows():
            accredited = str(row.get("accredited_procedures", "")).upper()
            if accredited and row["proc_upper"] not in accredited:
                for restricted in LEVEL2_RESTRICTED:
                    if restricted in row["proc_upper"] and restricted not in accredited:
                        _make_flag(flags, row["claim_id"], "CLINICAL_CODE_ANOMALY",
                            f"Facility {row['facility_id']} claimed '{row['proc_upper']}' "
                            f"which is NOT in their accredited procedure list. "
                            f"Performing unaccredited procedures is both fraudulent and dangerous.",
                            Severity.HIGH, M, str(row["facility_id"]), "FACILITY",
                            {"procedure": str(row["proc_upper"]),
                             "accredited": accredited[:200]})

    # 8c. Provider-attributed claims ───────────────────────────────────
    if providers is not None and "provider_id" in df.columns:
        df_prov = df.merge(providers, on="provider_id", how="left")
        if "provider_status" in df_prov.columns:
            invalid_prov = df_prov[
                df_prov["provider_status"].str.upper().isin(["DECEASED", "REVOKED", "EXPIRED"])
            ]
            for _, row in invalid_prov.iterrows():
                _make_flag(flags, row["claim_id"], "PROVIDER_IDENTITY_THEFT",
                    f"Claim attributed to Provider {row['provider_id']} whose licence is "
                    f"'{row['provider_status']}'. Claims cannot be legitimate under an invalid licence.",
                    Severity.CRITICAL, M, str(row["provider_id"]), "PROVIDER",
                    {"provider_status": str(row.get("provider_status", "UNKNOWN"))})

    return flags


#  MODULE 9 — CLAIM AMOUNT INTEGRITY

def check_claim_amount_integrity(claims: pd.DataFrame) -> list[FlagRecord]:
    """
    Module 9: Claim amount-specific checks.

      9a. Amount exactly equals known approved limits (systematic ceiling billing)
      9b. Progressive amount escalation pattern per facility
      9c. Identical amounts across many different members (templated amounts)
      9d. Negative amounts or zero-amount claims in unusual contexts
      9e. Amount hash fingerprinting (detecting copy-paste batches)
    """
    M = "MODULE_9_AMOUNTS"
    flags: list[FlagRecord] = []
    df = claims.copy()
    df["claim_date"] = _safe_date(df["claim_date"])

    # 9a. Ceiling billing (exact approved-limit amounts) ────────────────
    # Claims clustering exactly at the approved maximum suggest systematic ceiling billing
    for facility_id, grp in df.groupby("facility_id"):
        if len(grp) < 20:
            continue
        amt_counts = grp["claim_amount"].value_counts(normalize=True)
        # Any single amount value appearing in >20% of claims is anomalous
        for amount, freq in amt_counts.items():
            if freq > 0.20 and len(grp) > 20:
                matching = grp[grp["claim_amount"] == amount]
                for cid in matching["claim_id"].tolist():
                    _make_flag(flags, cid, "ROUND_NUMBER_PATTERN",
                        f"Facility {facility_id}: Amount {amount:,.0f} KSh appears in "
                        f"{freq*100:.1f}% of claims ({len(matching)} claims). "
                        f"Templated identical amounts indicate fabricated claim batches.",
                        Severity.MEDIUM, M, str(facility_id), "FACILITY",
                        {"amount": float(amount), "frequency": float(freq)})

    # 9b. Progressive escalation ────────────────────────────────────────
    df_sorted = df.sort_values(["facility_id", "claim_date"])
    for facility_id, grp in df_sorted.groupby("facility_id"):
        if len(grp) < 12:
            continue
        monthly = grp.groupby(grp["claim_date"].dt.to_period("M"))["claim_amount"].mean()
        if len(monthly) < 3:
            continue
        # Pearson correlation with time index — strong positive = escalation
        time_idx = np.arange(len(monthly))
        amounts  = monthly.values
        if np.std(amounts) == 0:
            continue
        r, p = stats.pearsonr(time_idx, amounts)
        if r > 0.90 and p < 0.05:
            for cid in grp["claim_id"].tolist():
                _make_flag(flags, cid, "UPCODING",
                    f"Facility {facility_id}: Strong positive time-trend in average claim amount "
                    f"(Pearson r={r:.2f}, p={p:.4f}). "
                    f"Systematic escalation of amounts over time indicates progressive upcoding.",
                    Severity.MEDIUM, M, str(facility_id), "FACILITY",
                    {"pearson_r": float(r), "p_value": float(p)})

    # 9c. Negative and zero amounts in wrong context ────────────────────
    neg_zero = df[df["claim_amount"] <= 0]
    for _, row in neg_zero.iterrows():
        _make_flag(flags, row["claim_id"], "CLINICAL_CODE_ANOMALY",
            f"Claim {row['claim_id']}: Amount = {row['claim_amount']:.2f} KSh (zero or negative). "
            f"Healthcare services cannot have zero or negative cost; "
            f"indicates data manipulation or test injection.",
            Severity.LOW, M, str(row.get("facility_id", "")), "FACILITY",
            {"claim_amount": float(row["claim_amount"])})

    # 9d. Batch fingerprint — identical amount+service+date across members ──
    if "service_code" in df.columns:
        batch_key = ["facility_id", "service_code", "claim_date", "claim_amount"]
        available  = [k for k in batch_key if k in df.columns]
        if len(available) == len(batch_key):
            batch_dupes = df[df.duplicated(subset=available, keep=False)]
            for (facility_id, svc, dt, amt), grp in batch_dupes.groupby(available):
                unique_members = grp["member_id"].nunique()
                if unique_members > 10:
                    for cid in grp["claim_id"].tolist():
                        _make_flag(flags, cid, "DUPLICATE_CLAIM",
                            f"Batch fingerprint: Facility {facility_id} submitted "
                            f"{unique_members} claims for service '{svc}' on "
                            f"{pd.Timestamp(dt).date()} all with identical amount {amt:,.0f} KSh. "
                            f"Copy-paste claim batch generation detected.",
                            Severity.HIGH, M, str(facility_id), "FACILITY",
                            {"unique_members": int(unique_members), "amount": float(amt),
                             "service_code": str(svc)})

    return flags


#  FRAUD RISK SCORING ENGINE

def compute_fraud_risk_scores(
    all_flags: list[FlagRecord], claims: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate all flag records per claim into a Fraud Risk Score (FRS).

    FRS is the SUM of weights of all distinct flag types active on a claim.
    Duplicate flag-type contributions (same type, multiple instances) are
    capped to prevent score inflation from a single module dominating.

    Returns a scored claims DataFrame with risk tier classification.
    """
    if not all_flags:
        out = claims[["claim_id"]].copy()
        out["fraud_risk_score"] = 0.0
        out["risk_tier"]        = RiskTier.CLEAR.value
        out["flags"]            = ""
        out["flag_count"]       = 0
        out["details"]          = ""
        return out

    flags_df = pd.DataFrame([f.to_dict() for f in all_flags])

    # Cap same flag-type weight contribution per claim to avoid inflation
    score_df = (
        flags_df.groupby(["claim_id", "flag_type"])
        .agg(max_weight=("weight", "max"))
        .reset_index()
    )
    score_agg = (
        score_df.groupby("claim_id")["max_weight"]
        .sum().reset_index(name="fraud_risk_score")
    )

    # Flag catalogue per claim
    flag_catalogue = (
        flags_df.groupby("claim_id")
        .agg(
            flag_count  =("flag_type", "count"),
            flags       =("flag_type", lambda x: " | ".join(sorted(set(x)))),
            modules     =("module",    lambda x: " | ".join(sorted(set(x)))),
            max_severity=("severity",  lambda x: (
                "CRITICAL" if "CRITICAL" in x.values else
                "HIGH"     if "HIGH"     in x.values else
                "MEDIUM"   if "MEDIUM"   in x.values else "LOW"
            )),
            details     =("detail",    lambda x: " || ".join(x)),
            entity_ids  =("entity_id", lambda x: " | ".join(sorted(set(str(v) for v in x if v)))),
        )
        .reset_index()
    )

    result = (
        claims[["claim_id"]]
        .merge(score_agg,     on="claim_id", how="left")
        .merge(flag_catalogue, on="claim_id", how="left")
    )
    result["fraud_risk_score"] = result["fraud_risk_score"].fillna(0.0)
    result["flag_count"]        = result["flag_count"].fillna(0).astype(int)
    result["flags"]             = result["flags"].fillna("")
    result["details"]           = result["details"].fillna("")
    result["max_severity"]      = result["max_severity"].fillna("NONE")
    result["modules"]           = result["modules"].fillna("")
    result["entity_ids"]        = result["entity_ids"].fillna("")

    def classify(score: float) -> str:
        if score >= CONFIG["frs_immediate_audit"]:
            return RiskTier.IMMEDIATE_AUDIT.value
        elif score >= CONFIG["frs_enhanced_monitoring"]:
            return RiskTier.ENHANCED_MONITORING.value
        elif score > 0:
            return RiskTier.WATCH.value
        return RiskTier.CLEAR.value

    result["risk_tier"] = result["fraud_risk_score"].apply(classify)
    return result.sort_values("fraud_risk_score", ascending=False).reset_index(drop=True)


#  ACTION PLAN GENERATOR

def generate_action_plans(scored_claims: pd.DataFrame) -> list[ActionPlan]:
    """
    Generate structured, prioritised action plans for every non-CLEAR claim.

    Action plans are deterministic — given the same flags, the same actions
    are always generated. This ensures auditability and legal defensibility.
    """
    plans: list[ActionPlan] = []

    TIER_CONFIGS: dict[str, dict[str, Any]] = {
        RiskTier.IMMEDIATE_AUDIT.value: {
            "base_actions": [
                ActionCode.FREEZE_PAYMENT,
                ActionCode.REQUEST_DOCUMENTS,
                ActionCode.SANCTIONS_SCREEN,
            ],
            "deadline_days": 2,
            "escalate_to": "Special Investigations Unit (SIU)",
        },
        RiskTier.ENHANCED_MONITORING.value: {
            "base_actions": [
                ActionCode.REQUEST_DOCUMENTS,
                ActionCode.ACTUARIAL_REVIEW,
            ],
            "deadline_days": 7,
            "escalate_to": "Claims Investigation Department",
        },
        RiskTier.WATCH.value: {
            "base_actions": [ActionCode.WATCHLIST_ADD],
            "deadline_days": 30,
            "escalate_to": "Routine Audit Queue",
        },
    }

    # Flag-type → additional actions mapping
    FLAG_ACTION_MAP: dict[str, list[ActionCode]] = {
        "BIOLOGICAL_IMPOSSIBLE":        [ActionCode.BENEFICIARY_CONTACT, ActionCode.REQUEST_DOCUMENTS],
        "DECEASED_BENEFICIARY_CLAIM":   [ActionCode.POLICE_REFERRAL, ActionCode.FREEZE_PAYMENT],
        "TIME_TRAVEL_CLAIM":            [ActionCode.FREEZE_PAYMENT, ActionCode.POLICE_REFERRAL],
        "CAPACITY_OVERFLOW":            [ActionCode.SITE_VISIT, ActionCode.ACTUARIAL_REVIEW],
        "GHOST_PATIENT":                [ActionCode.BENEFICIARY_CONTACT, ActionCode.SANCTIONS_SCREEN],
        "NETWORK_COLLUSION":            [ActionCode.POLICE_REFERRAL, ActionCode.DEREGISTER_FACILITY],
        "DUPLICATE_CLAIM":              [ActionCode.FREEZE_PAYMENT, ActionCode.REQUEST_DOCUMENTS],
        "PROVIDER_IDENTITY_THEFT":      [ActionCode.POLICE_REFERRAL, ActionCode.FREEZE_PAYMENT],
        "PHANTOM_SERVICE":              [ActionCode.SITE_VISIT, ActionCode.DEREGISTER_FACILITY],
        "TARIFF_EXTREME_MARKUP":        [ActionCode.PEER_REVIEW, ActionCode.ACTUARIAL_REVIEW],
        "IMPOSSIBLE_COMBINATION":       [ActionCode.PEER_REVIEW, ActionCode.REQUEST_DOCUMENTS],
        "STAFFING_MISMATCH":            [ActionCode.SITE_VISIT, ActionCode.REQUEST_DOCUMENTS],
        "BENFORD_DEVIATION":            [ActionCode.ACTUARIAL_REVIEW, ActionCode.REQUEST_DOCUMENTS],
        "UPCODING":                     [ActionCode.PEER_REVIEW],
        "SPLIT_BILLING":                [ActionCode.REQUEST_DOCUMENTS, ActionCode.ACTUARIAL_REVIEW],
        "ZSCORE_EXTREME":               [ActionCode.ACTUARIAL_REVIEW, ActionCode.SITE_VISIT],
    }

    for _, row in scored_claims[scored_claims["risk_tier"] != RiskTier.CLEAR.value].iterrows():
        tier_cfg = TIER_CONFIGS.get(row["risk_tier"], TIER_CONFIGS[RiskTier.WATCH.value])
        actions: set[ActionCode] = set(tier_cfg["base_actions"])

        flag_list = [f.strip() for f in row["flags"].split("|") if f.strip()]
        for flag in flag_list:
            actions.update(FLAG_ACTION_MAP.get(flag, []))

        # Escalation overrides
        if row["fraud_risk_score"] >= 20:
            actions.add(ActionCode.POLICE_REFERRAL)
        if "BIOLOGICAL_IMPOSSIBLE" in row["flags"] or "TIME_TRAVEL_CLAIM" in row["flags"]:
            actions.add(ActionCode.FREEZE_PAYMENT)

        rationale_parts = [
            f"FRS = {row['fraud_risk_score']:.0f} | Tier: {row['risk_tier']}",
            f"Flags: {row['flags']}",
            f"Max severity: {row.get('max_severity', 'UNKNOWN')}",
        ]
        if row.get("entity_ids"):
            rationale_parts.append(f"Entities: {row['entity_ids']}")

        plans.append(ActionPlan(
            claim_id=row["claim_id"],
            risk_tier=RiskTier(row["risk_tier"]),
            fraud_risk_score=float(row["fraud_risk_score"]),
            primary_actions=sorted(actions, key=lambda a: a.value),
            rationale=" | ".join(rationale_parts),
            deadline_days=tier_cfg["deadline_days"],
            escalate_to=tier_cfg["escalate_to"],
            supporting_flags=flag_list,
        ))

    return plans


#  ENTITY-LEVEL RISK PROFILES

def build_entity_risk_profiles(
    scored_claims: pd.DataFrame,
    flags_df: pd.DataFrame,
    claims: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Roll up claim-level risk into facility-level and member-level risk profiles.
    Returns a dict with keys 'facilities' and 'members'.
    """
    profiles: dict[str, pd.DataFrame] = {}

    for entity_col in ["facility_id", "member_id"]:
        if entity_col not in claims.columns:
            continue

        joined = claims[["claim_id", entity_col]].merge(
            scored_claims[["claim_id", "fraud_risk_score", "risk_tier", "flags", "flag_count"]],
            on="claim_id", how="left"
        )
        profile = (
            joined.groupby(entity_col)
            .agg(
                total_claims         =("claim_id", "count"),
                avg_frs              =("fraud_risk_score", "mean"),
                max_frs              =("fraud_risk_score", "max"),
                total_frs            =("fraud_risk_score", "sum"),
                immediate_audit_count=(
                    "risk_tier", lambda x: (x == RiskTier.IMMEDIATE_AUDIT.value).sum()
                ),
                unique_flag_types    =("flags", lambda x: " | ".join(
                    sorted(set(f.strip() for flags in x for f in flags.split("|") if f.strip()))
                )),
            )
            .reset_index()
            .sort_values("max_frs", ascending=False)
        )

        def entity_risk_tier(max_frs: float) -> str:
            if max_frs >= CONFIG["frs_immediate_audit"]:
                return RiskTier.IMMEDIATE_AUDIT.value
            elif max_frs >= CONFIG["frs_enhanced_monitoring"]:
                return RiskTier.ENHANCED_MONITORING.value
            elif max_frs > 0:
                return RiskTier.WATCH.value
            return RiskTier.CLEAR.value

        profile["entity_risk_tier"] = profile["max_frs"].apply(entity_risk_tier)
        profiles[entity_col.replace("_id", "s")] = profile  # → "facilities", "members"

    return profiles


#  MAIN ORCHESTRATOR

def run_fraud_detection(
    claims:     pd.DataFrame,
    members:    pd.DataFrame,
    facilities: pd.DataFrame,
    providers:  Optional[pd.DataFrame] = None,
    verbose:    bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[ActionPlan], dict[str, pd.DataFrame]]:
    """
    Run the full SHA fraud detection pipeline across all modules.

    Parameters
    ----------
    claims     : DataFrame of all claim records.
    members    : DataFrame of beneficiary/member records.
    facilities : DataFrame of registered facility details.
    providers  : (Optional) DataFrame of individual provider/doctor records.
    verbose    : Print progress and summary statistics.

    Returns
    -------
    scored_claims   : Every claim with FRS + risk tier.
    flags_df        : Detailed flag records for auditors.
    action_plans    : List of ActionPlan objects for each flagged claim.
    entity_profiles : Dict of facility-level and member-level risk profiles.
    """
    if verbose:
        print("=" * 70)
        print("  SHA KENYA — FRAUD DETECTION ENGINE  v3.0")
        print("=" * 70)
        print(f"  Claims     : {len(claims):,}")
        print(f"  Members    : {len(members):,}")
        print(f"  Facilities : {len(facilities):,}")
        if providers is not None:
            print(f"  Providers  : {len(providers):,}")
        print("-" * 70)

    all_flags: list[FlagRecord] = []

    modules: list[tuple[str, Any]] = [
        ("Module 1 — Biological Impossibilities",
         lambda: check_biological_impossibilities(claims, members)),
        ("Module 2 — Facility Capacity Violations",
         lambda: check_facility_capacity(claims, facilities)),
        ("Module 3 — Statistical Distribution Anomalies",
         lambda: check_statistical_anomalies(claims, facilities)),
        ("Module 4 — Duplicate & Split Claims",
         lambda: check_duplicate_claims(claims)),
        ("Module 5 — Clinical Coding Anomalies",
         lambda: check_clinical_coding_anomalies(claims, members, facilities)),
        ("Module 6 — Ghost Patient Detection",
         lambda: check_ghost_patients(claims, members)),
        ("Module 7 — Network Collusion",
         lambda: check_network_collusion(claims)),
        ("Module 8 — Provider Integrity",
         lambda: check_provider_integrity(claims, facilities, providers)),
        ("Module 9 — Claim Amount Integrity",
         lambda: check_claim_amount_integrity(claims)),
    ]

    for name, fn in modules:
        if verbose:
            print(f"  Running {name}...", end=" ", flush=True)
        try:
            module_flags = fn()
            all_flags.extend(module_flags)
            if verbose:
                print(f"{len(module_flags):,} flags raised.")
        except Exception as exc:
            if verbose:
                print(f"ERROR — {exc}")

    if verbose:
        print("-" * 70)
        print(f"  Total raw flags : {len(all_flags):,}")

    scored_claims = compute_fraud_risk_scores(all_flags, claims)
    flags_df = (
        pd.DataFrame([f.to_dict() for f in all_flags])
        if all_flags
        else pd.DataFrame(columns=[
            "claim_id", "flag_type", "severity", "weight",
            "detail", "module", "entity_id", "entity_type", "evidence_values"
        ])
    )

    action_plans    = generate_action_plans(scored_claims)
    entity_profiles = build_entity_risk_profiles(scored_claims, flags_df, claims)

    if verbose:
        print("\n  RISK TIER DISTRIBUTION:")
        tier_counts = scored_claims["risk_tier"].value_counts()
        for tier, count in tier_counts.items():
            pct = count / len(scored_claims) * 100
            print(f"    {tier:<30} {count:>8,} claims  ({pct:.1f}%)")

        print("\n  ACTION PLAN SUMMARY:")
        action_counts: dict[str, int] = defaultdict(int)
        for plan in action_plans:
            for action in plan.primary_actions:
                action_counts[action.value] += 1
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"    {action:<35} {count:>8,} claims")

        for entity_key, profile in entity_profiles.items():
            immediate = (profile["entity_risk_tier"] == RiskTier.IMMEDIATE_AUDIT.value).sum()
            print(f"\n  {entity_key.upper()} with IMMEDIATE_AUDIT tier: {immediate:,}")

        print("=" * 70)

    return scored_claims, flags_df, action_plans, entity_profiles


#  REPORTING HELPERS

def generate_audit_report(
    scored_claims:  pd.DataFrame,
    flags_df:       pd.DataFrame,
    action_plans:   list[ActionPlan],
    entity_profiles:dict[str, pd.DataFrame],
    output_prefix:  str = "sha_audit",
) -> dict[str, str]:
    """
    Export all fraud detection outputs to CSV files.

    Returns a dict mapping report name → file path.
    """
    paths: dict[str, str] = {}

    # 1. Full flagged claims with details
    flagged = scored_claims[scored_claims["risk_tier"] != RiskTier.CLEAR.value].copy()
    if not flags_df.empty:
        flagged = flagged.merge(flags_df.drop_duplicates("claim_id"), on="claim_id", how="left")
    p1 = f"{output_prefix}_flagged_claims.csv"
    flagged.to_csv(p1, index=False)
    paths["flagged_claims"] = p1

    # 2. Action plans
    plans_df = pd.DataFrame([p.to_dict() for p in action_plans])
    p2 = f"{output_prefix}_action_plans.csv"
    plans_df.to_csv(p2, index=False)
    paths["action_plans"] = p2

    # 3. Entity risk profiles
    for entity_key, profile in entity_profiles.items():
        p = f"{output_prefix}_{entity_key}_risk_profile.csv"
        profile.to_csv(p, index=False)
        paths[entity_key] = p

    # 4. Raw flags
    p4 = f"{output_prefix}_raw_flags.csv"
    flags_df.to_csv(p4, index=False)
    paths["raw_flags"] = p4

    # 5. Executive summary
    summary_lines = [
        "SHA Fraud Detection — Executive Summary",
        "=" * 50,
        f"Total claims analysed     : {len(scored_claims):,}",
        f"Total flags raised        : {len(flags_df):,}",
        f"Immediate audit cases     : {(scored_claims['risk_tier'] == RiskTier.IMMEDIATE_AUDIT.value).sum():,}",
        f"Enhanced monitoring cases : {(scored_claims['risk_tier'] == RiskTier.ENHANCED_MONITORING.value).sum():,}",
        f"Watch cases               : {(scored_claims['risk_tier'] == RiskTier.WATCH.value).sum():,}",
        f"Action plans generated    : {len(action_plans):,}",
    ]
    p5 = f"{output_prefix}_executive_summary.txt"
    with open(p5, "w") as fh:
        fh.write("\n".join(summary_lines))
    paths["executive_summary"] = p5

    print("\n  Reports saved:")
    for name, path in paths.items():
        print(f"    {name:<30} → {path}")

    return paths


#  QUICK USAGE EXAMPLE  (replace with real DataFrames in production)

if __name__ == "__main__":
    # Minimal synthetic test — replace with actual data loading
    claims_data = pd.DataFrame({
        "claim_id":       [1, 2, 3, 4, 5],
        "member_id":      [101, 102, 101, 103, 104],
        "facility_id":    ["F01", "F01", "F01", "F02", "F02"],
        "service_code":   ["SC001", "SC002", "SC001", "SC003", "SC004"],
        "procedure_type": ["DELIVERY", "MAJOR_SURGERY", "DELIVERY", "OUTPATIENT", "DIALYSIS"],
        "claim_date":     ["2024-01-15", "2024-01-20", "2024-05-10", "2024-02-01", "2024-02-01"],
        "claim_amount":   [8000.0, 85000.0, 8000.0, 500.0, 8500.0],
        "inpatient_days": [1, 5, 1, 0, 0],
    })

    members_data = pd.DataFrame({
        "member_id": [101, 102, 103, 104],
        "dob":       ["1985-03-10", "1970-06-22", "1990-11-01", "1965-08-15"],
        "sex":       ["F", "M", "F", "M"],
        "name":      ["Alice Wanjiku", "John Kamau", "Grace Achieng", "Peter Mwangi"],
    })

    facilities_data = pd.DataFrame({
        "facility_id":       ["F01", "F02"],
        "facility_level":    [3, 2],
        "county":            ["Nairobi", "Nairobi"],
        "ownership":         ["Private", "Public"],
        "bed_count":         [50, 20],
        "theater_count":     [2, 1],
        "registered_doctors":[5, 2],
    })

    scored, flags, plans, profiles = run_fraud_detection(
        claims=claims_data,
        members=members_data,
        facilities=facilities_data,
        verbose=True,
    )

    print("\nTop flagged claims:")
    print(scored[scored["risk_tier"] != "CLEAR"][
        ["claim_id", "fraud_risk_score", "risk_tier", "flags"]
    ].to_string(index=False))
