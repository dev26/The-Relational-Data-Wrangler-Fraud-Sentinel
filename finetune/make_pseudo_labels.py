#!/usr/bin/env python3
"""
Generate PSEUDO-LABELS for a LoRA fine-tuning fidelity experiment.

IMPORTANT — HONESTY NOTE:
  These labels are NOT verified fraud labels. No ground-truth fraud labels exist
  in the dataset. Each label here is simply the VERDICT PRODUCED BY THE RULE ENGINE
  in fraud_sentinel.py (rules + IsolationForest). Fine-tuning on them teaches the
  SLM to *imitate the rule engine*, and evaluation measures ONLY how well the tuned
  model reproduces the rule engine's verdicts on held-out rows (label-fidelity).
  It does NOT and cannot measure fraud-detection accuracy.

Outputs (MLX-LM "completions" format) into finetune/data/:
  train.jsonl, valid.jsonl, test.jsonl  -> {"prompt": "...", "completion": " LABEL"}
"""
import sys, json, random
from pathlib import Path

# reuse the baseline pipeline WITHOUT modifying it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fraud_sentinel as fs

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
random.seed(42)

def feature_prompt(row):
    """Compact, sanitized feature summary. Untrusted merchant text is passed
    through the same sanitizer used in production (no raw injection reaches the model)."""
    safe_merchant, _ = fs.sanitize(row.get("merchant_name"))
    def g(c, d="?"):
        v = row.get(c)
        return d if v is None or (isinstance(v, float) and v != v) else v
    feats = (
        f"amount={row['amount_filled']:.0f}; "
        f"amt_to_avg_ratio={g('amount_to_account_avg_ratio')}; "
        f"distance_km={g('distance_from_home_km')}; "
        f"hour={int(row['hour']) if row['hour']==row['hour'] else '?'}; "
        f"foreign={int(row.get('is_foreign_transaction_b',0))}; "
        f"new_device={int(row.get('is_new_device_b',0))}; "
        f"txn_24h={g('txn_count_last_24h')}; "
        f"status={row['status_clean']}; "
        f"risk_rating={g('risk_rating')}; "
        f"kyc={g('kyc_status')}; "
        f"merchant(untrusted)={safe_merchant}"
    )
    return (
        "You are a fraud triage model. Given the transaction features, respond with "
        "exactly one word: FRAUD or LEGITIMATE.\n"
        f"Features: {feats}\nLabel:"
    )

def main():
    m, report = fs.load_and_clean()
    df = fs.score(m)
    rows = []
    for _, r in df.iterrows():
        label = "FRAUD" if bool(r["is_fraud"]) else "LEGITIMATE"
        rows.append({"prompt": feature_prompt(r), "completion": f" {label}"})

    random.shuffle(rows)
    n = len(rows)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    splits = {
        "train": rows[:n_tr],
        "valid": rows[n_tr:n_tr + n_va],
        "test":  rows[n_tr + n_va:],
    }
    for name, part in splits.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            for ex in part:
                f.write(json.dumps(ex) + "\n")

    pos = sum(1 for r in rows if r["completion"].strip() == "FRAUD")
    print(json.dumps({
        "note": "PSEUDO-LABELS from rule engine, NOT verified fraud labels",
        "total": n, "train": len(splits["train"]),
        "valid": len(splits["valid"]), "test": len(splits["test"]),
        "pseudo_fraud_rate": round(pos / n, 4),
    }))

if __name__ == "__main__":
    main()
