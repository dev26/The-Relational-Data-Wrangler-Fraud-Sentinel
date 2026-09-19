#!/usr/bin/env python3
"""
Evaluate LoRA fine-tuning as a PSEUDO-LABEL FIDELITY experiment.

Compares, on the held-out test split:
  - base model (zero-shot)
  - base model + LoRA adapter (fine-tuned)
against the rule-engine PSEUDO-LABELS.

Reports overall agreement AND FRAUD-class precision/recall/F1, so the
class imbalance (~9% fraud) cannot hide behind a high accuracy number.

HONESTY: "fidelity" = agreement with the rule engine's own verdicts. It is NOT
fraud-detection accuracy; no verified fraud labels exist. We do NOT claim any
fraud-accuracy improvement.
"""
import json, sys, re
from pathlib import Path
from mlx_lm import load, generate

HERE = Path(__file__).resolve().parent
MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER = HERE / "adapters"
TEST = HERE / "data" / "test.jsonl"

def read_test():
    return [json.loads(l) for l in open(TEST)]

def predict(model, tok, prompt):
    out = generate(model, tok, prompt=prompt, max_tokens=4, verbose=False)
    up = out.upper()
    if "FRAUD" in up and "LEGIT" not in up:
        return "FRAUD"
    if "LEGIT" in up:
        return "LEGITIMATE"
    return "FRAUD" if "FRAUD" in up else "LEGITIMATE"

def metrics(gold, pred):
    tp = sum(g == "FRAUD" and p == "FRAUD" for g, p in zip(gold, pred))
    fp = sum(g != "FRAUD" and p == "FRAUD" for g, p in zip(gold, pred))
    fn = sum(g == "FRAUD" and p != "FRAUD" for g, p in zip(gold, pred))
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"agreement": round(acc, 4), "fraud_precision": round(prec, 4),
            "fraud_recall": round(rec, 4), "fraud_f1": round(f1, 4),
            "n": len(gold), "n_fraud_pseudo": sum(g == "FRAUD" for g in gold)}

def run(model, tok, data):
    gold = [ex["completion"].strip() for ex in data]
    pred = [predict(model, tok, ex["prompt"]) for ex in data]
    return metrics(gold, pred)

def main():
    data = read_test()
    print(f"loading base model {MODEL} ...", file=sys.stderr)
    base_model, tok = load(MODEL)
    base = run(base_model, tok, data)
    print(f"loading fine-tuned (LoRA adapter) ...", file=sys.stderr)
    ft_model, tok2 = load(MODEL, adapter_path=str(ADAPTER))
    ft = run(ft_model, tok2, data)

    report = {
        "experiment": "pseudo-label fidelity (LoRA), NOT fraud accuracy",
        "pseudo_label_source": "rule engine in fraud_sentinel.py (rules + IsolationForest)",
        "base_zero_shot": base,
        "fine_tuned_lora": ft,
        "delta_agreement": round(ft["agreement"] - base["agreement"], 4),
        "delta_fraud_f1": round(ft["fraud_f1"] - base["fraud_f1"], 4),
        "caveats": [
            "Labels are pseudo-labels (rule-engine verdicts), not verified fraud.",
            "Fidelity measures imitation of the rule engine, not fraud detection.",
            "~9% fraud base rate: judge by fraud_f1, not agreement alone.",
        ],
    }
    out = HERE / "eval_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
