#!/usr/bin/env python3
"""
Strict validator for fraud_output.jsonl.

Checks, per the assignment schema:
  - every line is valid JSON with EXACTLY the required keys
  - correct types: transaction_id:str, is_fraud:bool, confidence:float, justification:str
  - confidence in [0, 1]
  - non-empty justification
  - transaction_ids are unique
  - coverage == the set of UNIQUE transaction_ids in transactions.csv

Exits 0 if all pass, 1 otherwise. Prints a clear PASS/FAIL report.

Usage:  python3 validate_output.py [fraud_output.jsonl] [transactions.csv]
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = {"transaction_id", "is_fraud", "confidence", "justification"}

def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fraud_output.jsonl"
    txn_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "transactions.csv"

    failures = []           # (line_no, message)
    rows, ids = [], []

    with open(out_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                failures.append((i, f"invalid JSON: {e}"))
                continue
            rows.append(r)
            keys = set(r)
            if keys != REQUIRED:
                failures.append((i, f"keys {sorted(keys)} != required {sorted(REQUIRED)}"))
                continue
            if not isinstance(r["transaction_id"], str) or not r["transaction_id"]:
                failures.append((i, "transaction_id not a non-empty str"))
            if not isinstance(r["is_fraud"], bool):
                failures.append((i, f"is_fraud not bool: {type(r['is_fraud']).__name__}"))
            if not isinstance(r["confidence"], (int, float)) or isinstance(r["confidence"], bool):
                failures.append((i, "confidence not a number"))
            elif not (0.0 <= float(r["confidence"]) <= 1.0):
                failures.append((i, f"confidence out of [0,1]: {r['confidence']}"))
            if not isinstance(r["justification"], str) or not r["justification"].strip():
                failures.append((i, "justification not a non-empty str"))
            ids.append(r.get("transaction_id"))

    # uniqueness
    dup = {x for x in ids if ids.count(x) > 1} if len(ids) < 5000 else \
          {x for x, c in __import__("collections").Counter(ids).items() if c > 1}
    if dup:
        failures.append((0, f"{len(dup)} duplicate transaction_ids, e.g. {list(dup)[:3]}"))

    # coverage vs unique input ids
    coverage_ok = True
    try:
        import pandas as pd
        uniq = set(pd.read_csv(txn_path)["transaction_id"].dropna().unique())
        outset = set(ids)
        missing = uniq - outset
        extra = outset - uniq
        if missing:
            failures.append((0, f"{len(missing)} input txn_ids missing from output, e.g. {list(missing)[:3]}"))
            coverage_ok = False
        if extra:
            failures.append((0, f"{len(extra)} output txn_ids not in input, e.g. {list(extra)[:3]}"))
            coverage_ok = False
    except Exception as e:
        failures.append((0, f"coverage check skipped: {e}"))
        coverage_ok = False

    print("=" * 56)
    print(f"VALIDATE {out_path.name}: {len(rows)} records")
    n_fraud = sum(1 for r in rows if r.get("is_fraud") is True)
    print(f"  is_fraud=true: {n_fraud} ({n_fraud/max(len(rows),1):.1%})")
    print(f"  unique ids   : {len(set(ids))}/{len(ids)}")
    print(f"  coverage     : {'OK' if coverage_ok else 'MISMATCH'}")
    if failures:
        print(f"\nFAIL — {len(failures)} problem(s):")
        for ln, msg in failures[:25]:
            where = f"line {ln}" if ln else "global"
            print(f"  [{where}] {msg}")
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more")
        print("=" * 56)
        sys.exit(1)
    print("\nPASS — all schema/type/range/uniqueness/coverage checks OK")
    print("=" * 56)
    sys.exit(0)

if __name__ == "__main__":
    main()
