# SHA Kenya — Statistical Fraud Detection Engine
A Python-based fraud detection system for Kenya's **Social Health Authority (SHA)** that applies biological, physiological, operational, and statistical constraints to identify anomalous and potentially fraudulent insurance claims.

## Overview
Every legitimate medical event is constrained by biology, physics, time, and staffing. This engine encodes those constraints as testable rules and applies them systematically across a claims database — shifting fraud detection from reactive investigation to proactive, probabilistic screening.
The system flags the top percentage of claims most likely to be fraudulent and assigns each claim a **Fraud Risk Score (FRS)** for prioritised human audit.

## Features
| Module | What It Detects |
|||
| **Biological Impossibilities** | Same-day deliveries, deliveries within 9 months of each other, male obstetric claims, >3 major surgeries in one month, insufficient inter-surgery recovery |
| **Facility Capacity Violations** | Inpatient-days exceeding physical bed capacity, theater time overflow, doctor throughput ceiling breaches |
| **Statistical Anomalies** | Benford's Law digit manipulation, Z-score peer outliers, month-end claim stuffing, sudden volume spikes |
| **Duplicate Claims** | Exact duplicates and near-duplicates submitted within a 3-day window |
| **Clinical Coding Anomalies** | Sex-procedure mismatches, impossible same-day procedure combinations, Level-2 facilities billing specialist procedures |
| **Ghost Patient Detection** | Members with only high-cost claims and zero routine care, shared address/phone clusters, bulk agent enrolment patterns |
| **Network Collusion** | Members with >95% claim concentration at a single facility, agent-facility concentration ratios |

## Project Structure
```
sha_fraud/
├── sha_fraud_detector.py   # Core detection engine (import this in production)
├── sha_fraud_demo.py       # Demo runner with synthetic data + injected fraud
├── README.md               # This file
└── USAGE.md                # Detailed usage guide with code examples
```

## Quick Start
### 1. Install dependencies
```bash
pip install pandas numpy scipy
```
### 2. Run the demo
```bash
python sha_fraud_demo.py
```
This generates 2,000 synthetic claims, injects 8 known fraud patterns, runs the full detection pipeline, and saves an audit report to `sha_audit_report.csv`.
### 3. Use with your real data
```python
import pandas as pd
from sha_fraud_detector import run_fraud_detection, generate_audit_report
claims     = pd.read_csv("your_claims.csv")
members    = pd.read_csv("your_members.csv")
facilities = pd.read_csv("your_facilities.csv")
scored_claims, flags_df = run_fraud_detection(claims, members, facilities)
generate_audit_report(scored_claims, flags_df, "audit_output.csv")
```
See `USAGE.md` for full column specifications and configuration options.

## Risk Tier Classification
Every claim is assigned one of four risk tiers based on its cumulative Fraud Risk Score (FRS):
| Tier | FRS Threshold | Action |
||||
| `IMMEDIATE_AUDIT` | ≥ 15 | Escalate to audit unit immediately |
| `ENHANCED_MONITORING` | 8 – 14 | Place under enhanced review |
| `WATCH` | 1 – 7 | Log and monitor; include in periodic review |
| `CLEAR` | 0 | No flags raised |

## System Requirements
- Python 3.10 or higher
- pandas ≥ 1.5
- numpy ≥ 1.23
- scipy ≥ 1.9

## Data Requirements
Your input DataFrames must contain the following minimum columns:
**claims**
```
claim_id, member_id, facility_id, service_code,
procedure_type, claim_date, claim_amount, inpatient_days
```
**members**
```
member_id, sex, dob, registration_agent_id, registration_date
```
**facilities**
```
facility_id, facility_level, county, bed_count,
theater_count, registered_doctors, ownership
```
See `USAGE.md` for full schema details and optional enrichment columns.

## Configuration
All detection thresholds are defined in the `CONFIG` dictionary at the top of `sha_fraud_detector.py` and can be adjusted to match SHA's operational context:
```python
CONFIG = {
    "min_inter_delivery_days": 270,       # Minimum days between deliveries
    "max_major_surgeries_per_month": 3,   # Max major surgeries per patient/month
    "suspicious_occupancy_rate": 0.85,    # Bed occupancy soft-flag threshold
    "benford_pvalue_threshold": 0.05,     # Benford's Law significance level
    "frs_immediate_audit": 15,            # FRS score triggering immediate audit
    # ... and more
}
```

## Output
`run_fraud_detection()` returns two DataFrames:
**`scored_claims`** — one row per claim
| Column | Description |
|||
| `claim_id` | Original claim identifier |
| `fraud_risk_score` | Cumulative FRS (sum of all flag weights) |
| `risk_tier` | `IMMEDIATE_AUDIT` / `ENHANCED_MONITORING` / `WATCH` / `CLEAR` |
| `flag_count` | Number of distinct flags raised |
| `flags` | Pipe-separated list of flag types triggered |
**`flags_df`** — one row per individual flag raised
| Column | Description |
|||
| `claim_id` | Claim that triggered the flag |
| `flag_type` | Category of the anomaly (e.g. `BIOLOGICAL_IMPOSSIBLE`) |
| `severity` | `HIGH` or `MEDIUM` |
| `weight` | Numeric weight contributing to the FRS |
| `detail` | Human-readable explanation for auditors |

## Flag Weights Reference
| Flag Type | Weight | Severity |
||||
| `BIOLOGICAL_IMPOSSIBLE` | 10 | HIGH |
| `IMPOSSIBLE_COMBINATION` | 9 | HIGH |
| `CAPACITY_OVERFLOW` | 8 | HIGH |
| `ZSCORE_EXTREME` | 7 | HIGH |
| `NETWORK_COLLUSION` | 7 | HIGH |
| `DUPLICATE_CLAIM` | 6 | HIGH/MEDIUM |
| `GHOST_PATIENT` | 6 | HIGH/MEDIUM |
| `CLINICAL_CODE_ANOMALY` | 5 | HIGH |
| `BENFORD_DEVIATION` | 5 | MEDIUM |
| `STAFFING_MISMATCH` | 4 | HIGH |
| `TEMPORAL_CLUSTERING` | 4 | MEDIUM |
| `UPCODING` | 4 | MEDIUM |
| `STATISTICAL_OUTLIER` | 3 | MEDIUM |

## Licence
Developed for SHA Kenya — Fraud Analytics Unit. Internal use only.
