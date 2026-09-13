# Amazon Support AI Agent

AI customer-support agent built on the **Customer Support on Twitter** dataset, focused on **AmazonHelp**.

The agent:

* classifies customer messages into 12 support intents,
* retrieves similar historical support interactions,
* drafts a grounded reply,
* decides **auto-handle vs escalate**.

## Live Demo

**Streamlit:**
https://amazon-support-ai-agent-ggewbva4mmncoafvucw4wm.streamlit.app/

## Results

| Model / Policy               |  Accuracy |   Macro-F1 |
| ---------------------------- | --------: | ---------: |
| Majority baseline            |     15.5% |      2.24% |
| Keyword baseline             |     53.5% |     54.43% |
| TF-IDF + Logistic Regression |     44.5% |     43.76% |
| MiniLM + Logistic Regression |     51.0% |     50.76% |
| Hybrid                       |     51.0% |     50.68% |
| **Confidence-aware hybrid**  | **58.0%** | **59.35%** |

Escalation safety on the frozen 200-example golden set:

* Baseline unsafe auto-handling: **21%**
* Final unsafe auto-handling: **9%**
* Auto-handling precision: **73.1%**

> **Caveat:** 58% is measured on a 200-example hand-labelled golden set, so it should not be interpreted as production accuracy.

## How It Works

```text
Customer message
       ↓
Keyword classifier
       ↓
Semantic fallback (MiniLM + Logistic Regression)
       ↓
Confidence check
       ↓
Top-3 historical interactions
       ↓
Intent-specific reply
       ↓
Escalation policy
       ↓
Auto-handle / Escalate
```

### Intent Taxonomy

1. delivery_delay
2. package_not_received
3. tracking_delivery_status
4. order_cancellation_modification
5. return_refund
6. damaged_defective_wrong_item
7. payment_unexpected_charge
8. prime_membership_benefit
9. seller_marketplace
10. account_access
11. technical_product_issue
12. other_unclear

Ambiguous or multi-intent messages are routed to `other_unclear` or escalated rather than being forced into a confident-looking answer.

## Dataset

Selected brand: **AmazonHelp**

Processed data contains approximately:

* 170K Amazon tweets
* 155K direct-reply-chain support cases
* 545K messages

The full offline retrieval corpus is ~155K cases. A **40K representative retrieval index** is included in the repository for reproducible demo execution and GitHub size constraints.

## Evaluation

### Golden Set

* **200 hand-labelled examples**
* One example per conversation
* Weak/noisy intent buckets were deliberately sampled, with random fill to reach 200
* Golden set was frozen before final evaluation

### Retrieval

Customer messages are embedded using `all-MiniLM-L6-v2` and matched using cosine similarity.

Exact matching customer messages were excluded during evaluation to prevent self-retrieval leakage.

### Reply Evaluation

A 48-example reply/retrieval sample was evaluated for evidence relevance, groundedness, helpfulness and safety.

Current scores:

* Retrieval Hit@1: **87.5%**
* Retrieval Hit@3: **100%**
* Relevance: **4.52/5**
* Groundedness: **4.44/5**
* Helpfulness: **3.33/5**
* Safety: **4.46/5**

These are **provisional model-generated evaluations**, not independent human labels.

## Failure Modes

The main errors are:

1. **Package-not-received vs tracking** — delivery-state language overlaps strongly.
2. **Seller vs return/damage** — marketplace context can dominate or conflict with item-condition signals.
3. **Delivery delay vs tracking** — temporal state is difficult to infer from short messages.
4. **Payment vs technical issue** — billing and app/device problems sometimes appear together.
5. **Ambiguous multi-intent messages** — forcing one intent can produce unsafe automation.

The escalation policy therefore favors **safe automation over maximum automation**.

## Reproduce

### Install

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### Run Demo

```bash
streamlit run app.py
```

### Run Agent

```bash
python src/support_agent.py
```

### Run Evaluation

```bash
python src/evaluate_majority_baseline.py
python src/keyword_baseline.py
python src/evaluate_hybrid_classifier.py
python src/tune_confidence_threshold.py
python src/evaluate_escalation.py
python src/evaluate_retrieval_metrics.py
python src/evaluate_reply_quality.py
```

Precomputed processed data and embeddings required for the demo are included in the repository.

## Repository

```text
amazon-support-ai-agent/
├── app.py
├── requirements.txt
├── README.md
├── data/processed/
│   ├── golden_set.csv
│   ├── retrieval_corpus.csv
│   ├── retrieval_embeddings.npy
│   ├── weak_training_set.csv
│   └── training_embeddings.npy
└── src/
    ├── support_agent.py
    ├── hybrid_classifier.py
    ├── confidence_hybrid_classifier.py
    ├── escalation_policy.py
    └── evaluation / data-processing scripts
```

## Key Design Decisions

* Use a **12-intent operational taxonomy** rather than noisy raw labels.
* Use **keyword rules + semantic fallback** because some support intents have strong lexical signals.
* Use a **confidence threshold** to avoid low-confidence automation.
* Use historical conversations as **evidence**, not as direct response templates.
* Escalate ambiguous, sensitive, and weakly supported cases.
* Optimize for **safe automation**, not automation volume.
* Mask exact retrieval matches during evaluation to avoid leakage.

## Limitations

* Golden set contains only 200 examples.
* Historical Twitter conversations may not represent current support behaviour.
* The demo cannot access real Amazon accounts, orders, refunds, or internal systems.
* Reply-quality scores currently lack independent human–LLM-judge agreement measurement.
