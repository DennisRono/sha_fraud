# USAGE GUIDE — SHA Fraud Detection Engine

This guide covers everything you need to integrate the SHA Fraud Detection Engine
into a production analytics workflow, from data preparation to interpreting results.

## Table of Contents
1. [Installation](#1-installation)
2. [File Descriptions](#2-file-descriptions)
3. [Data Schema Reference](#3-data-schema-reference)
4. [Running the Demo](#4-running-the-demo)
5. [Using with Real SHA Data](#5-using-with-real-sha-data)
6. [Understanding the Output](#6-understanding-the-output)
7. [Adjusting Detection Thresholds](#7-adjusting-detection-thresholds)
8. [Running Individual Modules](#8-running-individual-modules)
9. [Exporting Audit Reports](#9-exporting-audit-reports)
10. [Interpreting Fraud Risk Scores](#10-interpreting-fraud-risk-scores)
11. [Common Errors and Fixes](#11-common-errors-and-fixes)
12. [Extending the Engine](#12-extending-the-engine)

## 1. Installation
### Requirements
- Python **3.10 or higher**
- pip (comes with Python)
### Install Python dependencies
Open a terminal or command prompt and run:
```bash
pip install pandas numpy scipy
```
Or if you are working in an environment where you need to avoid breaking system packages:
```bash
pip install pandas numpy scipy --break-system-packages
```
### Verify installation
```bash
python -c "import pandas, numpy, scipy; print('All dependencies ready.')"
```

## 2. File Descriptions
| File | Role |
|||
| `sha_fraud_detector.py` | Core detection library. Contains all 7 detection modules, the scoring engine, and the audit report generator. **This is the only file you need to import in production.** |
| `sha_fraud_demo.py` | Standalone demo. Generates synthetic SHA-like data, injects 8 fraud patterns, runs the full pipeline, and saves results to `sha_audit_report.csv`. Run this to verify the engine is working before connecting real data. |
| `README.md` | Project overview, quick start, and reference tables. |
| `USAGE.md` | This file — full integration guide. |

## 3. Data Schema Reference
The engine requires three DataFrames. Column names must match exactly (case-sensitive).

### 3a. `claims` DataFrame
One row per individual claim submitted to SHA.
| Column | Type | Required | Description |
|||||
| `claim_id` | string | **Yes** | Unique claim identifier |
| `member_id` | string | **Yes** | SHA beneficiary ID |
| `facility_id` | string | **Yes** | Facility that submitted the claim |
| `service_code` | string | **Yes** | ICD/CPT or SHA service code |
| `procedure_type` | string | **Yes** | Standardised procedure label (see below) |
| `claim_date` | string/date | **Yes** | Date of service (YYYY-MM-DD) |
| `claim_amount` | float | **Yes** | Amount claimed in KES |
| `inpatient_days` | int | **Yes** | Number of inpatient days (0 for outpatient) |
| `complexity` | int (1–5) | Optional | Consultation complexity level (enables upcoding detection) |
| `registration_agent_id` | string | Optional | Agent who registered the member (enables network analysis) |
#### Accepted `procedure_type` values
The engine matches these values (case-insensitive). Map your own codes to these labels
before running:
```
OUTPATIENT
MAJOR_SURGERY
MINOR_SURGERY
DELIVERY
C_SECTION
CAESAREAN
ICU
DIALYSIS
CHEMOTHERAPY
IMMUNIZATION
ANTENATAL
WELLNESS_VISIT
ROUTINE_CHECKUP
FAMILY_PLANNING
ORGAN_TRANSPLANT
APPENDECTOMY
COLONOSCOPY
HYSTERECTOMY
NEUROSURGERY
CARDIAC_BYPASS
ICU_VENTILATION
```

### 3b. `members` DataFrame
One row per registered SHA beneficiary.
| Column | Type | Required | Description |
|||||
| `member_id` | string | **Yes** | Unique member identifier (must match `claims`) |
| `sex` | string | **Yes** | `M` / `F` or `Male` / `Female` |
| `dob` | string/date | **Yes** | Date of birth (YYYY-MM-DD) |
| `registration_agent_id` | string | Optional | Agent who enrolled this member |
| `registration_date` | string/date | Optional | Date member was registered (enables ghost detection) |
| `address` | string | Optional | Member's physical address |
| `phone` | string | Optional | Member's phone number |
| `county` | string | Optional | County of residence |

### 3c. `facilities` DataFrame
One row per SHA-registered facility.
| Column | Type | Required | Description |
|||||
| `facility_id` | string | **Yes** | Unique facility identifier (must match `claims`) |
| `facility_level` | int | **Yes** | Kenya facility level: 2, 3, 4, 5, or 6 |
| `county` | string | **Yes** | County where facility is located (used for peer comparison) |
| `bed_count` | int | **Yes** | Number of registered inpatient beds |
| `theater_count` | int | **Yes** | Number of operating theaters |
| `registered_doctors` | int | **Yes** | Number of licensed doctors on register |
| `ownership` | string | Optional | `public` / `private` / `faith-based` |
| `facility_name` | string | Optional | Display name |

## 4. Running the Demo
The demo requires no external data. It generates everything internally.
```bash
python sha_fraud_demo.py
```
### What the demo does
1. Generates 20 synthetic facilities across Kenyan counties
2. Generates 500 synthetic members (mixed male/female, varied ages)
3. Generates 2,000 synthetic claims following realistic distributions
4. Injects 8 known fraud patterns:
   - Two deliveries on the same day (same member)
   - 30 major surgeries in one month (same member)
   - A male member claiming a delivery
   - A Level-2 facility with inpatient-days exceeding bed capacity
   - Month-end claim stuffing (75 claims in the last 5 days of a month)
   - Exact duplicate claim submission
   - A male member claiming a hysterectomy
   - Bulk ghost patient registration by a single agent
5. Runs all 7 detection modules
6. Prints a risk tier summary to the terminal
7. Saves `sha_audit_report.csv` to the working directory
### Expected terminal output
```
Generating synthetic SHA dataset...
Injecting fraud patterns...
Total claims (clean + fraudulent): 2,186
=================================================================
  SHA KENYA — FRAUD DETECTION ENGINE
=================================================================
  Claims   : 2,186
  Members  : 525
  Facilities: 20
--
  Running Module 1 — Biological Impossibilities... 155 flags raised.
  Running Module 2 — Facility Capacity Violations... 25 flags raised.
  Running Module 3 — Statistical Distribution Anomalies... 960 flags raised.
  Running Module 4 — Duplicate Claims... 220 flags raised.
  Running Module 5 — Clinical Coding Anomalies... 22 flags raised.
  Running Module 6 — Ghost Patient Detection... 135 flags raised.
  Running Module 7 — Network Collusion... 0 flags raised.
--
  Total flags raised: 1,517
  RISK TIER DISTRIBUTION:
    IMMEDIATE_AUDIT           110 claims  (5.0%)
    ENHANCED_MONITORING       194 claims  (8.9%)
    WATCH                     506 claims  (23.1%)
    CLEAR                   1,376 claims  (62.9%)
=================================================================
```

## 5. Using with Real SHA Data
### Step 1 — Load your data
```python
import pandas as pd
from sha_fraud_detector import run_fraud_detection, generate_audit_report
# Load from CSV exports
claims     = pd.read_csv("sha_claims_export.csv")
members    = pd.read_csv("sha_members_export.csv")
facilities = pd.read_csv("sha_facilities_register.csv")
```
### Step 2 — Standardise column names
If your column names differ from the schema above, rename them:
```python
claims = claims.rename(columns={
    "ClaimID":       "claim_id",
    "MemberNo":      "member_id",
    "FacilityCode":  "facility_id",
    "ServiceDate":   "claim_date",
    "ClaimAmount":   "claim_amount",
    "ProcedureName": "procedure_type",
    "IPDays":        "inpatient_days",
    "ICD10Code":     "service_code",
})
```
### Step 3 — Standardise procedure types
Map your internal procedure codes to the engine's expected labels:
```python
PROCEDURE_MAP = {
    "IP-DEL":    "DELIVERY",
    "IP-CS":     "C_SECTION",
    "OP-GEN":    "OUTPATIENT",
    "SURG-MAJ":  "MAJOR_SURGERY",
    "SURG-MIN":  "MINOR_SURGERY",
    "ICU-ADM":   "ICU",
    # ... add all your codes
}
claims["procedure_type"] = claims["procedure_type"].map(PROCEDURE_MAP).fillna(claims["procedure_type"])
```
### Step 4 — Ensure date formats
```python
claims["claim_date"]         = pd.to_datetime(claims["claim_date"])
members["dob"]               = pd.to_datetime(members["dob"])
members["registration_date"] = pd.to_datetime(members["registration_date"])
```
### Step 5 — Run the engine
```python
scored_claims, flags_df = run_fraud_detection(
    claims     = claims,
    members    = members,
    facilities = facilities,
    verbose    = True    # Set False to suppress terminal output
)
```
### Step 6 — Export results
```python
generate_audit_report(scored_claims, flags_df, output_path="sha_fraud_audit.csv")
```

## 6. Understanding the Output
`run_fraud_detection()` returns two DataFrames.
### `scored_claims` — one row per claim
```python
print(scored_claims.head())
#        claim_id  fraud_risk_score         risk_tier  flag_count                              flags
# 0     CLM90027                70  IMMEDIATE_AUDIT         4  BENFORD_DEVIATION | BIOLOGICAL_IMPOSSIBLE | ...
# 1     CLM90026                70  IMMEDIATE_AUDIT         4  BENFORD_DEVIATION | BIOLOGICAL_IMPOSSIBLE | ...
# ...
```
Filter to only actionable claims:
```python
# All claims requiring action
actionable = scored_claims[scored_claims["risk_tier"] != "CLEAR"]
# Only immediate audits
immediate = scored_claims[scored_claims["risk_tier"] == "IMMEDIATE_AUDIT"]
# Sort by risk score descending
top_risks = scored_claims.sort_values("fraud_risk_score", ascending=False)
```

### `flags_df` — one row per individual flag
```python
print(flags_df.head())
#       claim_id              flag_type severity  weight   detail
# 0   CLM90001  BIOLOGICAL_IMPOSSIBLE     HIGH      10   Member MBR0001 is MALE but claimed a delivery...
# 1   CLM90002       DUPLICATE_CLAIM   MEDIUM       6   Near-duplicate: Member MBR0010, service SVC500...
```
Filter flags by type or severity:
```python
# All biological impossibility flags
bio_flags = flags_df[flags_df["flag_type"] == "BIOLOGICAL_IMPOSSIBLE"]
# Only HIGH severity
high_flags = flags_df[flags_df["severity"] == "HIGH"]
# All flags for a specific claim
claim_detail = flags_df[flags_df["claim_id"] == "CLM90027"]
# Flag frequency breakdown
print(flags_df.groupby("flag_type")["claim_id"].count().sort_values(ascending=False))
```

## 7. Adjusting Detection Thresholds
All thresholds live in the `CONFIG` dictionary inside `sha_fraud_detector.py`.
Edit them to calibrate against SHA's actual operational context.
```python
# Option A: Edit sha_fraud_detector.py directly
CONFIG = {
    "min_inter_delivery_days": 270,        # Change to 240 if you want a tighter window
    "max_major_surgeries_per_month": 3,    # Raise to 5 for high-complexity hospitals
    "suspicious_occupancy_rate": 0.85,     # Lower to 0.75 for more aggressive flagging
    "frs_immediate_audit": 15,             # Raise to 20 to reduce immediate audit volume
    ...
}
# Option B: Override CONFIG at runtime before calling run_fraud_detection
import sha_fraud_detector as fd
fd.CONFIG["max_major_surgeries_per_month"] = 5
fd.CONFIG["frs_immediate_audit"] = 20
scored_claims, flags_df = fd.run_fraud_detection(claims, members, facilities)
```
### Threshold tuning guidance
| Parameter | Tighten (more flags) | Loosen (fewer flags) |
||||
| `suspicious_occupancy_rate` | Lower (e.g. 0.70) | Raise (e.g. 0.90) |
| `zscore_investigate_threshold` | Lower (e.g. 2.5) | Raise (e.g. 4.0) |
| `month_end_stuffing_ratio` | Lower (e.g. 0.30) | Raise (e.g. 0.50) |
| `frs_immediate_audit` | Lower (e.g. 10) | Raise (e.g. 20) |
| `benford_pvalue_threshold` | Raise (e.g. 0.10) | Lower (e.g. 0.01) |

## 8. Running Individual Modules
You can run any detection module in isolation for targeted analysis:
```python
from sha_fraud_detector import (
    check_biological_impossibilities,
    check_facility_capacity,
    check_statistical_anomalies,
    check_duplicate_claims,
    check_clinical_coding_anomalies,
    check_ghost_patients,
    check_network_collusion,
    compute_fraud_risk_scores,
)
import pandas as pd
claims     = pd.read_csv("claims.csv")
members    = pd.read_csv("members.csv")
facilities = pd.read_csv("facilities.csv")
# Example: run only biological checks
bio_flags = check_biological_impossibilities(claims, members)
print(f"{len(bio_flags)} biological impossibility flags raised.")
for flag in bio_flags[:5]:   # Preview first 5
    print(f"  [{flag['flag_type']}] {flag['detail']}")
# Example: run only duplicate detection
dup_flags = check_duplicate_claims(claims)
print(f"{len(dup_flags)} duplicate flags raised.")
# Combine selected modules and score
all_flags = bio_flags + dup_flags
scored = compute_fraud_risk_scores(all_flags, claims)
print(scored[scored["risk_tier"] != "CLEAR"])
```

## 9. Exporting Audit Reports
### Default CSV export
```python
from sha_fraud_detector import generate_audit_report
generate_audit_report(scored_claims, flags_df, output_path="sha_fraud_audit.csv")
```
The CSV contains one row per flagged claim with all flag details merged in.
### Custom export — Excel with risk tier tabs
```python
import pandas as pd
tiers = ["IMMEDIATE_AUDIT", "ENHANCED_MONITORING", "WATCH"]
with pd.ExcelWriter("sha_fraud_audit.xlsx", engine="openpyxl") as writer:
    for tier in tiers:
        subset = scored_claims[scored_claims["risk_tier"] == tier]
        subset.to_excel(writer, sheet_name=tier, index=False)
print("Excel report saved to sha_fraud_audit.xlsx")
```
### Custom export — top-N highest risk claims
```python
top50 = scored_claims.nlargest(50, "fraud_risk_score")
top50 = top50.merge(flags_df, on="claim_id", how="left")
top50.to_csv("top50_high_risk_claims.csv", index=False)
```

## 10. Interpreting Fraud Risk Scores
The Fraud Risk Score (FRS) is the **sum of weights of all flags raised** against a claim.
A single claim can accumulate flags from multiple modules simultaneously.
### Example score breakdown
```
Claim CLM90027 — FRS = 70
  BIOLOGICAL_IMPOSSIBLE   weight=10   (30 major surgeries in one month)
  DUPLICATE_CLAIM         weight= 6   (same member + service within 3 days)
  BENFORD_DEVIATION       weight= 5   (facility's claim amounts fail Benford's test)
  STATISTICAL_OUTLIER     weight= 3   (facility Z-score = 4.2 vs peer group)
  GHOST_PATIENT           weight= 6   (no preventive care, only high-cost procedures)
  ─────────────────────────────────────
  Total FRS               = 70   →  IMMEDIATE_AUDIT
```
### What to do with each tier
**`IMMEDIATE_AUDIT` (FRS ≥ 15)**
Suspend payment pending manual review. Assign to an auditor within 48 hours.
These claims have triggered multiple high-weight flags or at least one biological/physical impossibility.
**`ENHANCED_MONITORING` (FRS 8–14)**
Do not auto-approve. Route to secondary review queue. Request clinical documentation from the facility before processing payment.
**`WATCH` (FRS 1–7)**
Approve with logging. Include in monthly analytical review. Accumulating WATCH flags over time on the same facility is itself a signal.
**`CLEAR` (FRS = 0)**
No anomalies detected. Process normally.

## 11. Common Errors and Fixes
### `KeyError: 'procedure_type'`
Your claims DataFrame is missing the `procedure_type` column.
Rename the relevant column before calling the engine (see Step 3 in Section 5).
### `ValueError: could not convert string to float`
The `claim_amount` column contains non-numeric values (e.g. `"KES 5,000"` or empty strings).
Clean it first:
```python
claims["claim_amount"] = (
    claims["claim_amount"]
    .astype(str)
    .str.replace("KES", "")
    .str.replace(",", "")
    .str.strip()
)
claims["claim_amount"] = pd.to_numeric(claims["claim_amount"], errors="coerce").fillna(0)
```
### `KeyError: 'bed_count'`
Your facilities DataFrame is missing required capacity columns.
The engine needs `bed_count`, `theater_count`, and `registered_doctors` to run Module 2.
If this data is not yet available, run only the modules that don't need it:
```python
from sha_fraud_detector import check_biological_impossibilities, check_duplicate_claims
```
### Benford's Law module raises no flags
This module requires at least 100 claims per facility to produce a reliable result.
Facilities with fewer than 100 claims are skipped automatically.
### Module 7 (Network Collusion) raises 0 flags
Network collusion detection requires `registration_agent_id` to be present in both
the `claims` and `members` DataFrames. If this column is absent, the module runs
but finds nothing to analyse.
### Dates are not being parsed correctly
Ensure all date columns are in `YYYY-MM-DD` format or pass them as pandas Timestamps.
```python
claims["claim_date"] = pd.to_datetime(claims["claim_date"], dayfirst=True)
# Or for ambiguous formats:
claims["claim_date"] = pd.to_datetime(claims["claim_date"], format="%d/%m/%Y")
```

## 12. Extending the Engine
### Adding a new detection rule
Each detection module is a standalone function that takes DataFrames and returns a list
of flag dictionaries. To add a new rule, append to an existing module or create a new one
following this pattern:
```python
def check_my_new_rule(claims: pd.DataFrame, members: pd.DataFrame) -> list:
    flags = []
    # ... your detection logic ...
    for _, row in suspicious_claims.iterrows():
        flag_record(
            flags,
            claim_id  = row["claim_id"],
            flag_type = "MY_NEW_FLAG",          # Must be in FLAG_WEIGHTS dict
            detail    = f"Explanation for auditor: {row['claim_id']} ...",
            severity  = "HIGH"                  # or "MEDIUM"
        )
    return flags
```
Then register the weight in `FLAG_WEIGHTS`:
```python
FLAG_WEIGHTS = {
    # ... existing entries ...
    "MY_NEW_FLAG": 6,     # Choose a weight consistent with severity
}
```
And add the module to the `modules` list inside `run_fraud_detection()`:
```python
modules = [
    # ... existing modules ...
    ("Module 8 — My New Rule", lambda: check_my_new_rule(claims, members)),
]
```
### Adding new impossible procedure combinations
Edit the `IMPOSSIBLE_COMBINATIONS` list in `sha_fraud_detector.py`:
```python
IMPOSSIBLE_COMBINATIONS = [
    # (procedure_A, procedure_B, reason_for_auditor)
    ("APPENDECTOMY", "COLONOSCOPY", "Cannot perform colonoscopy prep and appendectomy same day"),
    # Add your combinations:
    ("CARDIAC_BYPASS", "DISCHARGE_SAME_DAY", "Cardiac bypass requires minimum 5-day post-op stay"),
    ("DIALYSIS", "KIDNEY_TRANSPLANT", "Transplant and dialysis not clinically indicated same day"),
]
```

### Scheduling automated runs
To run the engine on a schedule (e.g. nightly on new claims):

```python
import pandas as pd
from sha_fraud_detector import run_fraud_detection, generate_audit_report
from datetime import date
# Load only claims submitted today
claims     = pd.read_csv("claims.csv")
members    = pd.read_csv("members.csv")
facilities = pd.read_csv("facilities.csv")
today_claims = claims[claims["claim_date"] == str(date.today())]
scored, flags = run_fraud_detection(today_claims, members, facilities, verbose=False)
output_file = f"audit_{date.today()}.csv"
generate_audit_report(scored, flags, output_path=output_file)
print(f"Nightly audit complete → {output_file}")
```

*SHA Kenya — Fraud Analytics Unit | Internal Use Only*
