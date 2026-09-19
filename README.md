# The Relational Data Wrangler & Fraud Sentinel

Cleans and merges three relational datasets (`customers`, `accounts`, `transactions`),
scores each transaction for fraud, and emits **strict JSON per transaction**.

```json
{"transaction_id":"TXN_0000974","is_fraud":true,"confidence":0.5,"justification":"One sentence."}
```

## Run

```bash
python3 fraud_sentinel.py            # full run, uses the SLM for justifications
python3 fraud_sentinel.py --no-llm   # skip the SLM, deterministic template justifications
python3 fraud_sentinel.py --limit 50 # first 50 rows (smoke test)
```

Output → `fraud_output.jsonl` (one JSON object per line).

## Verify the submission (helper scripts, read-only, no retraining)

```bash
python3 test_pipeline.py       # injection defense + verdict independence + fallback tests
python3 data_quality.py        # writes data_quality_report.json + prints a summary
python3 validate_output.py     # strict schema/type/range/uniqueness/coverage check of the JSONL
```

- `test_pipeline.py` — adversarial prompt-injection cases (fake `SYSTEM:` instructions,
  `</untrusted_merchant>` delimiter escape, `<system>` token-fusion, brace/JSON payloads,
  unicode/zero-width), and proves the **verdict/confidence never change** when merchant
  text is replaced with hostile strings. Also checks the SLM is <3B and that the
  Ollama-down fallback works. Exit 0 = all pass.
- `data_quality.py` — machine-readable + text report: duplicates removed, invalid/missing
  values, orphaned relationships, final record counts, injection-flagged strings, and
  flagged-fraud counts (a risk score, **not** ground truth).
- `validate_output.py` — fails loudly (exit 1) on any bad key/type/range, duplicate id,
  or coverage mismatch; pass a path to validate a different JSONL.

## Dashboard (local, no dependencies)

```bash
python3 dashboard.py        # reads the existing files, writes dashboard.html
open dashboard.html         # macOS — or double-click it
# if open is blocked, serve it:  python3 -m http.server 8501
#                                then visit http://localhost:8501/dashboard.html
```

`dashboard.py` builds a **self-contained `dashboard.html`** from the real project files
(read-only). It shows an overview (total / flagged / flag-rate / verdict counts, computed
dynamically), a transaction explorer (search by ID, filter by verdict and confidence, sort
by any column), per-transaction details with justification and merged account/customer
info, and the data-quality metrics from `data_quality_report.json`. It **does not** run
Ollama, retrain, or recompute any verdict — values are shown exactly as produced by the
pipeline. It handles missing/empty/malformed inputs gracefully.

> Streamlit is not used: it fails to import in this project's Python env (a protobuf
> descriptor incompatibility). The static HTML build has zero runtime dependencies and is
> the reliable option. The dashboard restates that **risk flags are not confirmed fraud**
> and **confidence is not a calibrated probability**.

## Design decisions (why it is built this way)

The provided data has **no fraud labels** and **no `notes` column**, so:

- **Verdict = rules + IsolationForest, not a learned classifier.** Interpretable risk
  rules (amount-vs-average, new device + foreign, distance from home, odd hour,
  velocity, failed/reversed, high-risk/PEP/KYC) carry 65% of the score; an
  unsupervised `IsolationForest` anomaly score carries 35%. `is_fraud` is the
  thresholded combination; `confidence` is the distance from the decision boundary.
  This is reliable without ground truth and fully explainable.
- **The SLM only writes the one-sentence justification** — it never decides the
  verdict. Model: `qwen2.5:1.5b-instruct` (**1.5B < 3B**, open-weight, via Ollama).
  Output is length-validated; on any failure it falls back to a deterministic template,
  so the JSON is always valid.
- **Fine-tuning was deliberately skipped:** with no labels, no `torch`/`MLX` installed,
  and a 90-minute budget, LoRA fine-tuning would be high-risk for no accuracy gain.
  See "Stretch path" below.

## Data cleaning / wrangling

- **Duplicates:** drops 12 exact-duplicate transaction rows; dedupes IDs in all tables.
- **Corrupt amounts:** 22 non-numeric + 9 non-positive → flagged (`amount_invalid`) and
  median-imputed for modelling.
- **Invalid timestamps:** 90 unparseable → flagged (`ts_invalid`); hour derived from
  the reliable `transaction_hour`, falling back to the parsed timestamp.
- **Messy `status`:** `'  SUCCESS '`, `'success'`, `'Success'`, NaN … → normalized.
- **Booleans:** `Y/N`, `TRUE/FALSE`, `1/0`, mixed case → unified.
- **Joins (no row multiplication):** `transactions → accounts → customers` as left joins
  (M:1 each), merged rows stay at 988. `account.customer_id` is treated as source of
  truth (8 transactions had a mismatched/orphan `customer_id`).

## Prompt-injection defense (notes treated as untrusted)

There is no `notes` field; the only untrusted free-text is `merchant_name` (and
`merchant_id`, `ip_address`, …). Before any of it reaches the SLM prompt it is:

1. NFKC-normalized; control/zero-width characters stripped.
2. **All `<`/`>` removed** so it cannot escape the `<untrusted_merchant>` delimiter.
3. Injection tokens redacted (`ignore previous`, `system:/assistant:`, `is/not fraud`,
   `set is_fraud`, `confidence:`, code fences, `jailbreak`, `override`, …).
4. Truncated to 60 chars and wrapped in an explicit untrusted delimiter, with the
   model instructed to never follow instructions inside it.

**Crucially, the verdict never depends on free text** — injection can at most affect
one wording of a sentence, never `is_fraud`/`confidence`.

## Environment notes

- Python 3.13 (anaconda), pandas / numpy / scikit-learn, Apple M2 / 16 GB, Ollama.
- The script blocks the machine's broken `pyarrow` dylib at import time (a local
  anaconda protobuf-symbol clash) so scikit-learn imports cleanly — no env changes.

## Fine-tuning experiment (run, isolated, honest — see `finetune/`)

A small MLX LoRA experiment was run **without touching the baseline**. It is a
**pseudo-label fidelity** test, not a fraud-accuracy claim.

- **Pseudo-labels:** the rule engine's own verdicts on 988 txns (NOT verified fraud).
  Split 790 train / 98 valid / 100 test (`finetune/data/`).
- **Model:** `Qwen2.5-1.5B-Instruct-4bit` + LoRA (8 layers, 2.6M params, 200 iters,
  peak mem 2.7 GB). Adapter: `finetune/adapters/adapters.safetensors`.
- **Env note:** anaconda's MPICH makes MLX's default distributed init abort; the
  training driver forces the single-host `ring` backend to work around it.
- **Result (`finetune/eval_report.json`), reported honestly:**

  | | agreement | fraud precision | fraud recall | fraud F1 |
  |---|---|---|---|---|
  | base (zero-shot) | 0.27 | 0.11 | 0.75 | 0.20 |
  | fine-tuned LoRA | 0.88 | 0.00 | 0.00 | **0.00** |

  **The fine-tune FAILED on the objective.** The 0.88 "agreement" is majority-class
  collapse (88 legit / 100) — it predicts LEGITIMATE for everything and catches 0/12
  fraud cases. On the imbalanced pseudo-labels (~9% fraud), LoRA learned the majority
  class. **No fraud-accuracy improvement is claimed; the opposite occurred.**
- **Fix for future work (not run, out of timebox):** balance the training set
  (oversample fraud to ~40–50%) or use class weighting, then re-evaluate.

**Conclusion:** the production deliverable remains the rule+anomaly pipeline
(`fraud_sentinel.py` → `fraud_output.jsonl`). The fine-tune is a documented,
reproducible negative result, kept isolated in `finetune/`.

## Reproduce

```bash
python3 fraud_sentinel.py                         # baseline -> fraud_output.jsonl
python3 finetune/make_pseudo_labels.py            # regenerate pseudo-label splits
python3 finetune/train_driver.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data finetune/data --mask-prompt --num-layers 8 --batch-size 4 \
  --iters 200 --learning-rate 1e-4 --max-seq-length 256 --adapter-path finetune/adapters
python3 finetune/evaluate.py                       # -> finetune/eval_report.json
```
