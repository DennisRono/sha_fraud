import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sha_fraud_detector import run_fraud_detection, generate_audit_report
np.random.seed(42)
random.seed(42)
def random_date(start="2023-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return s + timedelta(days=random.randint(0, (e - s).days))
def make_facilities(n=20):
    levels = [2]*5 + [3]*7 + [4]*5 + [5]*2 + [6]*1
    counties = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
                "Machakos", "Nyeri", "Meru", "Kakamega", "Garissa"]
    rows = []
    for i in range(1, n+1):
        level = random.choice(levels)
        beds = {2: 30, 3: 80, 4: 200, 5: 400, 6: 800}[level]
        rows.append({
            "facility_id": f"FAC{i:03d}",
            "facility_name": f"{'Hospital' if level >= 4 else 'Clinic'} {i}",
            "facility_level": level,
            "county": random.choice(counties),
            "ownership": random.choice(["public", "private", "faith-based"]),
            "bed_count": beds + random.randint(-10, 10),
            "theater_count": max(1, level - 2),
            "registered_doctors": max(1, (level - 1) * 3 + random.randint(-1, 3)),
        })
    return pd.DataFrame(rows)
def make_members(n=500):
    rows = []
    agents = [f"AGT{i:03d}" for i in range(1, 11)]
    for i in range(1, n+1):
        sex = random.choice(["M", "F"])
        age = random.randint(5, 80)
        dob = datetime.today() - timedelta(days=age * 365)
        rows.append({
            "member_id": f"MBR{i:04d}",
            "sex": sex,
            "dob": dob.strftime("%Y-%m-%d"),
            "county": random.choice(["Nairobi", "Mombasa", "Kisumu", "Nakuru"]),
            "address": f"P.O. Box {random.randint(100,9999)}, Nairobi",
            "phone": f"07{random.randint(10000000, 99999999)}",
            "registration_agent_id": random.choice(agents),
            "registration_date": random_date("2022-01-01", "2023-06-01"),
        })
    return pd.DataFrame(rows)
PROCEDURE_TYPES = [
    "OUTPATIENT", "OUTPATIENT", "OUTPATIENT",
    "MINOR_SURGERY", "MAJOR_SURGERY",
    "DELIVERY", "C_SECTION",
    "ICU", "DIALYSIS", "IMMUNIZATION",
    "ANTENATAL", "WELLNESS_VISIT",
]
def make_claims(members, facilities, n=2000):
    rows = []
    for i in range(1, n+1):
        member = members.sample(1).iloc[0]
        facility = facilities.sample(1).iloc[0]
        proc = random.choice(PROCEDURE_TYPES)
        if proc in ["DELIVERY", "C_SECTION", "ANTENATAL"] and member["sex"] == "M":
            proc = "OUTPATIENT"
        date = random_date()
        inpatient_days = random.randint(1, 5) if proc in ["MAJOR_SURGERY", "ICU", "DELIVERY"] else 0
        rows.append({
            "claim_id": f"CLM{i:05d}",
            "member_id": member["member_id"],
            "facility_id": facility["facility_id"],
            "service_code": f"SVC{random.randint(100, 999)}",
            "procedure_type": proc,
            "claim_date": date.strftime("%Y-%m-%d"),
            "claim_amount": round(np.random.lognormal(8, 1.2), 2),
            "inpatient_days": inpatient_days,
            "complexity": random.choice([1, 2, 3, 4, 5]),
            "registration_agent_id": member["registration_agent_id"],
        })
    return pd.DataFrame(rows)

#  INJECT FRAUD PATTERNS

def inject_fraud(claims, members, facilities):
    """Inject known fraud scenarios into the dataset."""
    fraud_claims = []
    start_id = 90001
    def next_id():
        nonlocal start_id
        cid = f"CLM{start_id:05d}"
        start_id += 1
        return cid
    # Pick a female member for obstetric fraud
    female_member = members[members["sex"] == "F"].iloc[0]["member_id"]
    male_member = members[members["sex"] == "M"].iloc[0]["member_id"]
    facility = facilities.iloc[0]["facility_id"]
    agent = "AGT001"
    # ── FRAUD 1: Two deliveries on same day ───────────────────────────
    for _ in range(2):
        fraud_claims.append({
            "claim_id": next_id(), "member_id": female_member,
            "facility_id": facility, "service_code": "SVC200",
            "procedure_type": "DELIVERY", "claim_date": "2024-03-15",
            "claim_amount": 25000, "inpatient_days": 2, "complexity": 3,
            "registration_agent_id": agent,
        })
    # ── FRAUD 2: 30 major surgeries on one member in one month ────────
    for day in range(1, 16):
        for _ in range(2):
            fraud_claims.append({
                "claim_id": next_id(), "member_id": female_member,
                "facility_id": facility, "service_code": "SVC300",
                "procedure_type": "MAJOR_SURGERY", "claim_date": f"2024-06-{day:02d}",
                "claim_amount": 80000, "inpatient_days": 3, "complexity": 5,
                "registration_agent_id": agent,
            })
    # ── FRAUD 3: Male claiming delivery ──────────────────────────────
    fraud_claims.append({
        "claim_id": next_id(), "member_id": male_member,
        "facility_id": facility, "service_code": "SVC200",
        "procedure_type": "DELIVERY", "claim_date": "2024-07-10",
        "claim_amount": 25000, "inpatient_days": 2, "complexity": 3,
        "registration_agent_id": agent,
    })
    # ── FRAUD 4: Capacity overflow — Level-2 facility claiming 9999 inpatient-days ──
    small_fac = facilities[facilities["facility_level"] == 2].iloc[0]["facility_id"]
    for i in range(50):
        fraud_claims.append({
            "claim_id": next_id(), "member_id": f"MBR{(i+1):04d}",
            "facility_id": small_fac, "service_code": "SVC100",
            "procedure_type": "OUTPATIENT", "claim_date": f"2024-08-{random.randint(1,28):02d}",
            "claim_amount": 5000, "inpatient_days": 10, "complexity": 1,
            "registration_agent_id": agent,
        })
    # ── FRAUD 5: Month-end stuffing ───────────────────────────────────
    stuffing_fac = facilities.iloc[3]["facility_id"]
    for day in [26, 27, 28, 29, 30]:
        for _ in range(15):
            fraud_claims.append({
                "claim_id": next_id(), "member_id": f"MBR{random.randint(1,100):04d}",
                "facility_id": stuffing_fac, "service_code": "SVC111",
                "procedure_type": "OUTPATIENT", "claim_date": f"2024-09-{day:02d}",
                "claim_amount": 3500, "inpatient_days": 0, "complexity": 2,
                "registration_agent_id": agent,
            })
    # ── FRAUD 6: Exact duplicate claims ──────────────────────────────
    dupe_base = {
        "claim_id": next_id(), "member_id": "MBR0010",
        "facility_id": facility, "service_code": "SVC500",
        "procedure_type": "MINOR_SURGERY", "claim_date": "2024-05-20",
        "claim_amount": 12000, "inpatient_days": 1, "complexity": 2,
        "registration_agent_id": agent,
    }
    fraud_claims.append(dupe_base.copy())
    dupe_copy = dupe_base.copy()
    dupe_copy["claim_id"] = next_id()
    fraud_claims.append(dupe_copy)
    # ── FRAUD 7: Male claiming hysterectomy ───────────────────────────
    fraud_claims.append({
        "claim_id": next_id(), "member_id": male_member,
        "facility_id": facility, "service_code": "SVC601",
        "procedure_type": "HYSTERECTOMY", "claim_date": "2024-04-01",
        "claim_amount": 95000, "inpatient_days": 5, "complexity": 5,
        "registration_agent_id": agent,
    })
    # ── FRAUD 8: Bulk agent registration (ghost patients) ─────────────
    bulk_agent = "AGT009"
    bulk_reg_date = "2023-03-01"
    for j in range(25):
        mid = f"GHOST{j:03d}"
        members.loc[len(members)] = {
            "member_id": mid, "sex": "F", "dob": "1990-01-01",
            "county": "Nairobi", "address": "P.O. Box 999, Nairobi",
            "phone": "0700000001", "registration_agent_id": bulk_agent,
            "registration_date": bulk_reg_date,
        }
        fraud_claims.append({
            "claim_id": next_id(), "member_id": mid,
            "facility_id": facility, "service_code": "SVC700",
            "procedure_type": "MAJOR_SURGERY", "claim_date": "2024-01-10",
            "claim_amount": 120000, "inpatient_days": 5, "complexity": 5,
            "registration_agent_id": bulk_agent,
        })
    fraud_df = pd.DataFrame(fraud_claims)
    combined = pd.concat([claims, fraud_df], ignore_index=True)
    return combined, members

#  RUN THE DEMO

if __name__ == "__main__":
    print("\nGenerating synthetic SHA dataset...")
    facilities = make_facilities(20)
    members    = make_members(500)
    claims     = make_claims(members, facilities, n=2000)
    print("Injecting fraud patterns...")
    claims, members = inject_fraud(claims, members, facilities)
    print(f"Total claims (clean + fraudulent): {len(claims):,}\n")
    scored_claims, flags_df = run_fraud_detection(claims, members, facilities, verbose=True)
    print("\n  TOP 15 HIGHEST-RISK CLAIMS:")
    print("-" * 65)
    top = scored_claims[scored_claims["risk_tier"] != "CLEAR"].head(15)
    for _, row in top.iterrows():
        print(f"  [{row['risk_tier']:<22}] {row['claim_id']}  FRS={int(row['fraud_risk_score']):<4}  Flags: {row['flags']}")
    generate_audit_report(scored_claims, flags_df, "sha_audit_report.csv")
    print("\n  FLAG TYPE BREAKDOWN:")
    if not flags_df.empty:
        breakdown = flags_df.groupby(["flag_type", "severity"]).size().reset_index(name="count")
        breakdown = breakdown.sort_values("count", ascending=False)
        for _, row in breakdown.iterrows():
            print(f"    {row['flag_type']:<35} [{row['severity']:<6}]  {row['count']:>5} flags")
    print()
