# TEKNOBUILT AI-ML Engineering Assessment: Construction Data Automation

## Repository Structure

```text
teknobuilt_assessment/
├── README.md                    # Architectural documentation (Task 1 Resilience + Task 2 Architecture)
├── task1_extract_forms.py       # Task 1: CV Gatekeeper + VLM Structured Extraction Pipeline
├── extracted_forms.json         # Task 1: Validated JSON output for Sample_input.pdf
├── task2_p6_classifier_poc.py   # Task 2: Optional POC demonstrating deduplication & strict Enum schema
└── requirements.txt             # Python dependencies
```

## Quick Start

```bash
pip install -r requirements.txt

# Run Task 1 in offline mock mode (executes full OpenCV gatekeeper + Pydantic validation)
python task1_extract_forms.py --pdf Sample_input.pdf --template template.jpg --output extracted_forms.json

# Run Task 1 with live OpenAI GPT-4o Vision API
export OPENAI_API_KEY="sk-..."
python task1_extract_forms.py --pdf Sample_input.pdf --template template.jpg --live

# Run Task 2 POC against the P6 schedule sample
python task2_p6_classifier_poc.py --input P6_Schedule_Sample.xlsx
```

---

## Task 1: Field Data Digitization Pipeline — Resilience Strategy

Field safety forms are physical documents subject to skew, mud, folds, crossed-out ink, and corrupted scans. Sending raw pages directly to a Vision-Language Model (VLM) causes two failure modes: wasted API spend on garbage pages and hallucinated JSON on blank/corrupted inputs.

Our pipeline enforces a **3-stage defense-in-depth architecture**:

1. **Ingestion & Heuristic Pre-Filter (Zero-Cost):**
   - Rejects text-only error pages (e.g., Page 4 `"FAILED TEST FORM"` in `Sample_input.pdf`) or pages with missing image streams before any CV or VLM compute is triggered.
2. **Computer Vision Quality & Structural Gatekeeper (OpenCV):**
   - **Ink Density Bounds:** Computes foreground pixel ratio via binary thresholding. Pages with `<1%` ink (blank) or `>65%` ink (heavily muddy/blacked-out scans) are rejected immediately.
   - **Blur Detection:** Computes Laplacian variance; pages below the sharpness threshold (`<40.0`) are flagged for human rescanning rather than risking misread numbers/dates.
   - **Template Alignment (ORB + Homography):** Matches ORB keypoints against `template.jpg`. Pages failing the minimum structural match threshold (`<20` good matches) are rejected as non-PACE documents. Valid skewed pages are perspective-warped upright via RANSAC homography to maximize checkbox alignment accuracy.
3. **Strict Schema Validation & Dead-Letter Queue (DLQ):**
   - VLM outputs are constrained via Pydantic `response_format` (Structured Outputs) with `temperature=0.0`.
   - Rejected or low-confidence pages are routed to a structured DLQ (`status: "REJECTED_INVALID_FORM"`) with diagnostic metrics (`ink_ratio`, `blur_score`, `template_matches`) so downstream analytics pipelines never crash or ingest fabricated safety records.

---

## Task 2: P6 Schedule Batch Classification — Architecture Document

### Exploratory Analysis of `P6_Schedule_Sample.xlsx`
Inspecting the provided Primavera P6 schedule reveals three data characteristics that directly inform our architecture:
- **High Redundancy:** Out of **4,499** total rows, there are only **739** unique normalized `Activity Name` strings (**83.6% deduplication**).
- **Noisy Field Text & Encoding Artifacts:** Column headers contain UTF-8 BOM artifacts (`ï»¿Activity ID`), and activity names contain field typos (e.g., `"secuirty"`, `"Copmound"`).
- **Context-Dependent Generic Names:** High-frequency names like `"Testing"` (137 rows), `"Dressing"` (128 rows), and `"Labeling"` (128 rows) are ambiguous in isolation. However, their `Activity ID` prefixes (e.g., `TELE-C1-c1-6750` vs. `CON-C5-6970`) provide crucial discipline disambiguation.

---

### 1. Scalable Batch Processing Pipeline Architecture

To classify 50,000+ Primavera P6 rows in minutes rather than hours, we decouple ingestion from inference using an asynchronous **Deduplicate-Route-Batch-Join** architecture:

```text
[P6 CSV / XLSX (50,000+ rows)]
   │
   ▼ (1. Sanitize & Normalize: strip BOM, lowercase, extract Activity ID prefix)
[Deduplication Layer] ──► Compresses 50,000 rows to ~6,000–8,000 unique (Prefix, Activity Name) keys
   │
   ▼ (2. Tiered Classification Router)
   ├──► Tier 1: Exact Cache & Deterministic Regex Rules (Redis/Postgres) ──► ~35% resolved (<5ms, $0)
   ├──► Tier 2: Semantic Embedding Classifier (BGE-small + SetFit / kNN) ──► ~45% resolved (high confidence, ~$0)
   └──► Tier 3: Async Micro-Batched LLM (50 items/req, 20 workers)       ──► ~20% ambiguous tail
   │
   ▼ (3. Join back to all 50,000 Activity IDs + Confidence Audit Log)
[Validated Classified P6 Schedule]
```

- **Micro-Batching:** Instead of 1 row per API call, Tier 3 packs **50 unique activities per prompt** (indexed `0..49`), reducing HTTP overhead and repeated system-prompt tokens by **50x**.
- **Async Concurrency & Backpressure:** Workers execute via `asyncio` with a token-bucket rate limiter and exponential backoff for `429` responses. For non-urgent overnight schedule imports, **OpenAI / Anthropic Batch APIs** provide an additional **50% cost discount**.

---

### 2. Strict Constraint Enforcement (Zero Hallucinations)

To guarantee the AI strictly selects from the provided taxonomy (12 disciplines, **32 valid Discipline/Sub-Discipline pairs**) and never invents new categories or invalid cross-combinations (e.g., `Plumbing | Cable Pulling`):

1. **Flattened Pair Enum via Constrained Decoding (Structured Outputs):**
   We flatten the two-level hierarchy into a single `Literal` Enum of the exact 32 valid `"Discipline | Sub-Discipline"` strings (see `ValidTaxonomyPair` in `task2_p6_classifier_poc.py`). Passing this Pydantic schema to `response_format` (OpenAI/Gemini) or grammar-constrained decoding (`xgrammar` / `Outlines` on open-weights models) masks out invalid tokens at sampling time, making hallucinations mathematically impossible.
2. **Deterministic Post-Validation & Isolation:**
   Every batch response is validated against Pydantic to confirm (a) every input `batch_idx` (`0..N-1`) is accounted for and (b) every label belongs to the canonical 32-pair set. Any malformed or missing row is isolated and retried individually without failing the entire batch.

---

### 3. Cost & Accuracy Optimization Strategy

| Optimization Layer | Mechanism | Cost / Latency Impact | Accuracy Impact |
| :--- | :--- | :--- | :--- |
| **1. Key Deduplication** | Group by `(Activity_ID_Prefix, normalized_name)` before inference. | **-80% to -85%** inference volume (verified: 4,499 → 739 unique names in sample). | Neutral (guarantees 100% consistent labels for identical tasks). |
| **2. Context Injection** | Pass `Activity ID` prefix (e.g., `TELE-C1`) alongside `Activity Name`. | Negligible (+3 tokens/row). | **High gain** on generic activities like `"Testing"`, `"Labeling"`, `"Dressing"`. |
| **3. Tiered Model Cascade** | Route high-confidence items (`>=0.85`) through a local embedding classifier (`bge-small` + SetFit/LR); escalate only ambiguous/messy rows to a fast LLM (`gpt-4o-mini` / `gemini-2.5-flash`). | **-90%+** total API spend vs. running every row through a frontier reasoning model. | Maintains **>95% F1** while reserving LLM reasoning for construction slang and typos. |
| **4. Prompt Caching & Batching** | Pack 50 items/call behind a static cached system prompt + Batch API for offline schedules. | **-50% to -75%** on remaining LLM token costs. | Neutral; allows rich few-shot construction examples in the cached system prompt. |
| **5. Human-in-the-Loop (HITL) Cache** | Persist human-verified classifications in Postgres/Redis keyed by `(prefix, norm_name)`. | Recurring project schedules hit cache at **$0** cost. | Continuously compounds accuracy across projects. |
