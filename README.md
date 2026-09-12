# Amazon Customer Support AI Agent

A reproducible AI support-agent prototype for the **Hiver SDE Intern take-home assignment**.

The system:
- classifies customer messages into **12 operational intents**,
- retrieves similar historical AmazonHelp interactions as evidence,
- decides **auto-handle vs escalate** using a conservative safety policy,
- drafts a support reply without requiring an external LLM API.

> **Headline result:** 58.0% accuracy / 59.35% macro-F1 on a frozen 200-example golden set.  
> The final conservative escalation policy reduced unsafe auto-handling from **21% to 9%** of the golden set.

---

## 1. How It Works

```text
Customer message
       |
       v
Intent routing
(keyword rules -> MiniLM + Logistic Regression)
       |
       v
Predicted intent
       |
       +----------------------+
       |                      |
       v                      v
Historical retrieval     Safety policy
~155K AmazonHelp cases   auto-handle / escalate
       |                      |
       v                      v
Top-3 evidence          Conservative reply
```

The system deliberately separates **classification, retrieval/grounding, and safety routing** so failures are easier to inspect.

---

## 2. Dataset & Intent Taxonomy

Dataset: Kaggle **Customer Support on Twitter** (`thoughtvector/customer-support-on-twitter`).

I selected **AmazonHelp** because it has a large volume of support interactions and diverse operational scenarios.

The pipeline indexed approximately **2.8M tweets**, identified ~170K Amazon tweets, and built ~155K Amazon customer/support cases.

> Historical exchanges are treated as support interactions/evidence, **not guaranteed resolved cases**.

### 12 intents

| Intent | Meaning |
|---|---|
| `delivery_delay` | Expected delivery date has passed / order is late |
| `package_not_received` | Package was not received, including marked-delivered cases |
| `tracking_delivery_status` | Shipment status, location or ETA question without clear lateness |
| `order_cancellation_modification` | Cancel or modify an order |
| `return_refund` | General return/refund request |
| `damaged_defective_wrong_item` | Damaged, defective or incorrect item |
| `payment_unexpected_charge` | Unexpected, duplicate or disputed payment/charge |
| `prime_membership_benefit` | Prime membership or benefit issue |
| `seller_marketplace` | Marketplace seller issue |
| `account_access` | Account login/access issue |
| `technical_product_issue` | Product/technical functionality issue |
| `other_unclear` | Ambiguous, unsupported or unclear request |

Important boundaries:
- A known late order is `delivery_delay`, even if tracking is mentioned.
- "Marked delivered but not received" is `package_not_received`.
- Tracking/location/ETA without clear lateness is `tracking_delivery_status`.
- Damaged/wrong/defective product takes precedence over a generic return/refund request.
- Prime mention alone does not make an issue a Prime intent.
- Multiple or unclear issues may be `other_unclear`.

---

## 3. Intent Classification

The classifier uses:

1. **Keyword rules** for high-signal, interpretable cases.
2. **MiniLM (`all-MiniLM-L6-v2`) + Logistic Regression** as a semantic fallback.
3. A confidence threshold of **0.40**; lower-confidence semantic predictions become `other_unclear`.

The threshold was selected on the frozen golden set because it gave the best macro-F1:

| Threshold | Accuracy | Macro-F1 |
|---:|---:|---:|
| **0.40** | **0.580** | **0.5935** |
| 0.45 | 0.575 | 0.5924 |
| 0.50 | 0.540 | 0.5668 |
| 0.60 | 0.515 | 0.5465 |
| 0.80 | 0.435 | 0.4663 |

### Classification results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline | 0.155 | 0.0224 |
| Keyword baseline | 0.535 | 0.5443 |
| TF-IDF + Logistic Regression | 0.445 | 0.4376 |
| MiniLM + Logistic Regression | 0.510 | 0.5076 |
| **Confidence-aware hybrid** | **0.580** | **0.5935** |

The final model improves macro-F1 from **54.43% to 59.35%** over the keyword-only baseline.

---

## 4. Historical Retrieval

The retrieval corpus contains approximately **155K AmazonHelp customer/support exchanges**.

For a new message:
- the customer text is embedded with MiniLM,
- cosine similarity is calculated against historical customer messages,
- the top 3 interactions are returned as evidence.

During reply evaluation, the current golden-set case is masked to prevent leakage.

> **Important:** retrieval similarity is not answer quality. A similarity of 0.80 means the customer messages are semantically close; it does not mean the answer is 80% correct.

---

## 5. Reply Generation

The core demo does **not** require an external LLM API.

For supported intents, the agent uses conservative intent-specific templates informed by historical support behavior. Historical replies are shown as evidence rather than copied verbatim.

Twitter-specific artifacts such as old `@username` mentions, stale `t.co` URLs and agent initials are sanitized when historical fallback wording is used.

Sensitive payment/account cases are escalated instead of making account-specific guesses.

---

## 6. Escalation & Safety

The policy escalates cases when signals include:
- `other_unclear`,
- explicit human-support requests,
- multiple issue groups,
- account-access issues,
- payment/unexpected-charge issues,
- weak historical evidence,
- additional safety checks from the combined policy.

### Golden-set result

| Outcome | Count |
|---|---:|
| Auto-handled | 67 |
| Escalated | 133 |
| Safe auto-handled | 49 |
| Unsafe auto-handled | 18 |

This gives:
- **73.1% precision among auto-handled cases**
- **9.0% unsafe auto-handling across the full golden set**
- **24.5% safe automation rate**

These are evaluation-set measurements, not production guarantees.

---

## 7. Evaluation & Failure Modes

### Golden set

The evaluation set contains **200 manually labelled examples**, sampled using weak-bucket coverage plus random fill, and frozen before model evaluation.

### Top failure modes

**1. Delivery delay vs tracking**

> "Where is my order? It was supposed to arrive yesterday."

The classifier can choose tracking even though the operational intent is delivery delay.

**2. Package not received vs tracking**

Messages about packages marked delivered can contain strong tracking/status language, which can dominate keyword rules.

**3. Payment vs technical**

Payment issues may be described through technical symptoms such as missing payment emails or application behavior.

**4. Seller vs product/return**

One message can mention a seller problem, wrong product, damage and refund together. A single-label taxonomy cannot represent every multi-intent case.

**5. Ambiguous/out-of-scope messages**

A semantically similar message can still be operationally unclear. Low confidence, weak evidence and sensitive intents are therefore routed toward escalation.

### What is misleading about the headline number?

The **59.35% macro-F1** should not be interpreted as production-ready classification.

Main caveats:
- only 200 manually labelled evaluation examples,
- overlapping operational intents,
- weakly labelled training data,
- `other_unclear` absorbs ambiguous/out-of-scope cases,
- conservative escalation trades automation coverage for safety,
- retrieval similarity is not direct response-quality measurement.

Likewise, **73.1% auto-handle precision** means 73.1% of selected auto-handled cases were judged safe/appropriate under this evaluation protocol. It does not mean 73.1% of all customer requests can safely be automated.

---

## 8. Reproducibility / How to Run

### Requirements

Python 3.10+ recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first model run may download `all-MiniLM-L6-v2` from Hugging Face.

### Run the agent

From the repository root:

```powershell
python src\support_agent.py
```

Enter any customer message. The CLI prints:
- predicted intent,
- classification method and confidence,
- auto-handle/escalate decision,
- escalation reason,
- draft reply,
- top-3 historical evidence.

Example:

```text
My package says delivered but I never received it.
```

### Reproduce classifier experiments

```powershell
python src\tune_confidence_threshold.py
python src\evaluate_majority_baseline.py
python src\keyword_baseline.py
python src\train_tfidf_classifier.py
python src\train_semantic_classifier.py
```

**Fast reproduction:** the repository includes the processed training/retrieval artifacts used by the demo, so the main agent can be run without rebuilding the full raw dataset or retrieval embeddings.

---

## 9. Repository Structure

```text
HiverAssignment/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── processed/
│       ├── golden_set.csv
│       ├── weak_training_set.csv
│       ├── training_embeddings.npy
│       ├── retrieval_corpus.csv
│       ├── retrieval_embeddings.npy
│       └── evaluation artifacts
│
└── src/
    ├── support_agent.py              # main entry point
    ├── hybrid_classifier.py
    ├── confidence_hybrid_classifier.py
    ├── escalation_policy.py
    ├── data preparation scripts
    ├── evaluation scripts
    └── archive/                      # exploratory/intermediate scripts
```

The raw dataset and local virtual environment are not required for the main demo.

---

## 10. Decision Log

1. Selected AmazonHelp for volume and scenario diversity.
2. Chose a compact 12-intent taxonomy.
3. Added `other_unclear` rather than forcing unsupported messages into known intents.
4. Used clustering only for exploration because unsupervised separation was weak.
5. Created and froze a 200-example manually labelled golden set.
6. Kept the keyword baseline because it was strong and interpretable.
7. Compared TF-IDF and semantic classifiers.
8. Added confidence-aware semantic fallback.
9. Selected the 0.40 threshold using macro-F1.
10. Built retrieval over historical support interactions.
11. Added retrieval leakage protection.
12. Escalated payment/account issues conservatively.
13. Optimized escalation for safety, not maximum automation.
14. Sanitized historical Twitter artifacts.
15. Avoided an external LLM API so the core prototype remains locally reproducible.

---

## 11. Next Week

- Expand human-labelled training data, especially minority intents.
- Improve delivery-state boundaries.
- Add multi-intent detection and a dedicated out-of-scope detector.
- Build a genuinely human-labelled retrieval relevance set.
- Improve intent-aware evidence ranking.
- Add independent human reply-quality evaluation.
- Calibrate confidence and measure the automation-vs-safety curve.
- Test on a temporally separated holdout set.

---

## 12. Limitations

This is a take-home prototype, not a production customer-support system.

Known limitations:
- weakly supervised classifier training,
- only 200 manually labelled intent-evaluation examples,
- overlapping operational intents,
- no external LLM in the core reply layer,
- conservative escalation reduces automation coverage,
- historical Twitter behavior may not reflect current policy,
- no live order/account information is available.

The system is therefore evaluated on **behavioral correctness, safety, reproducibility and transparent failure analysis**, rather than production-scale automation claims.
