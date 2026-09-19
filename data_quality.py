#!/usr/bin/env python3
"""
Data-quality report generator (read-only; reuses fraud_sentinel's cleaning).

Produces a concise, machine-readable report of the wrangling step plus flagged
transaction counts, WITHOUT modifying the baseline pipeline or its output.

Outputs:
  data_quality_report.json   (machine-readable)
  prints a human-readable text summary to stdout

Usage:  python3 data_quality.py
"""
import json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fraud_sentinel as fs   # reuse cleaning/scoring/sanitizer; do not modify it

KEY_TXN_FIELDS = ["transaction_id", "account_id", "customer_id", "transaction_timestamp",
                  "amount", "currency", "transaction_type", "channel", "status",
                  "merchant_name", "merchant_category", "auth_method", "device_type",
                  "ip_address", "amount_to_account_avg_ratio", "time_since_prev_txn_mins"]

def main():
    # raw counts (pre-clean) for transparency
    raw_c = pd.read_csv(HERE / "customers.csv")
    raw_a = pd.read_csv(HERE / "accounts.csv")
    raw_t = pd.read_csv(HERE / "transactions.csv")

    # run the SAME cleaning + scoring the baseline uses (read-only)
    m, report = fs.load_and_clean()
    scored = fs.score(m)

    # enrich: per-field missingness on raw transactions (top offenders)
    miss = raw_t[[c for c in KEY_TXN_FIELDS if c in raw_t.columns]].isna().sum()
    miss = {k: int(v) for k, v in miss.items() if v > 0}

    # injection-flagged merchant strings (via the production sanitizer)
    inj_flagged = int(sum(fs.sanitize(v)[1] for v in raw_t.get("merchant_name", pd.Series([], dtype=object))))

    n = len(scored)
    n_fraud = int(scored["is_fraud"].sum())

    dq = {
        "record_counts": {
            "customers_raw": len(raw_c), "customers_dedup": int(raw_c["customer_id"].nunique()),
            "accounts_raw": len(raw_a), "accounts_dedup": int(raw_a["account_id"].nunique()),
            "transactions_raw": len(raw_t),
            "transactions_unique_ids": int(raw_t["transaction_id"].nunique()),
            "transactions_final": report["rows_final"],
        },
        "duplicates_removed": {
            "txn_full_duplicate_rows": report["txn_full_dups_dropped"],
            "txn_duplicate_ids_after": report["txn_id_dups_dropped"],
        },
        "invalid_or_corrupt": {
            "amount_non_numeric": report["amount_uncoercible"],
            "amount_non_positive": report["amount_non_positive"],
            "timestamp_unparseable": report["ts_unparseable"],
            "timestamp_in_future": report["ts_future"],
        },
        "orphaned_relationships": {
            "txn_account_id_not_in_accounts": report["txn_orphan_account"],
            "txn_customer_id_not_in_customers": report["txn_orphan_customer"],
            "note": "account.customer_id is treated as source of truth for merges",
        },
        "missing_values_by_field": miss,
        "prompt_injection": {
            "merchant_strings_flagged": inj_flagged,
            "note": "untrusted text is sanitized before reaching the SLM; it never affects the verdict",
        },
        "flagged_transactions": {
            "is_fraud_true": n_fraud,
            "is_fraud_false": n - n_fraud,
            "fraud_rate": round(n_fraud / n, 4),
            "note": "risk-based verdict (rules + IsolationForest); NO ground-truth labels exist",
        },
    }

    out = HERE / "data_quality_report.json"
    with open(out, "w") as f:
        json.dump(dq, f, indent=2)

    # text summary
    rc = dq["record_counts"]
    print("=" * 56)
    print("DATA-QUALITY SUMMARY")
    print("=" * 56)
    print(f"customers : {rc['customers_raw']} raw -> {rc['customers_dedup']} unique")
    print(f"accounts  : {rc['accounts_raw']} raw -> {rc['accounts_dedup']} unique")
    print(f"txns      : {rc['transactions_raw']} raw -> {rc['transactions_final']} final "
          f"({dq['duplicates_removed']['txn_full_duplicate_rows']} dup rows removed)")
    inv = dq["invalid_or_corrupt"]
    print(f"invalid   : amount(non-numeric={inv['amount_non_numeric']}, "
          f"non-positive={inv['amount_non_positive']}), "
          f"timestamp(unparseable={inv['timestamp_unparseable']}, future={inv['timestamp_in_future']})")
    orp = dq["orphaned_relationships"]
    print(f"orphans   : txn->account {orp['txn_account_id_not_in_accounts']}, "
          f"txn->customer {orp['txn_customer_id_not_in_customers']}")
    print(f"injection : {inj_flagged} merchant string(s) flagged & neutralized")
    ft = dq["flagged_transactions"]
    print(f"flagged   : {ft['is_fraud_true']} fraud / {n} ({ft['fraud_rate']:.1%})  "
          f"[risk score, not ground truth]")
    print("-" * 56)
    print(f"top missing fields: "
          + ", ".join(f"{k}={v}" for k, v in sorted(miss.items(), key=lambda x: -x[1])[:6]))
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
