# CRS Preference-Shift Evaluation — Living Methodology (Project Guardrails)

**Purpose of this document**  
This Markdown file is the **single source of truth** for (i) what the system is supposed to do, (ii) what each module is responsible for, (iii) the data contracts that must remain stable, and (iv) the conceptual definitions that prevent “metric drift” or thesis inconsistencies.

> **Update rule:** Whenever you change code, **update this file in the same commit**.  
> **Non‑negotiable:** Preserve the global schemas unless you also update all downstream consumers.

---

## 1) System Goal (Methodology Summary)

This project evaluates **prompted conversational recommender systems (CRS)** under **controlled preference shifts** in a **closed-domain movie item set (item universe / product assortment)**. It provides:

1. A **simulation harness** that generates multi-turn dialogues between:
   - a CRS agent (system) constrained to recommend from the fixed item set, and
   - a user simulator (user) that introduces **controlled preference shifts**.
2. A **grounded, interpretable evaluation suite**:
   - **Turn-level alignment metrics** (concept overlap, constraint similarity, copying penalty)
   - **Conversation-level adaptation metrics** (mean turn score + shift recovery diagnostics)
   - A single reported **Adaptation Score** for ranking, plus the components for diagnosis.
3. An optional **LLM-as-judge** rubric for convergent validity checks (not ground truth).

---

## 2) Thesis-safe Terminology (Do Not Mix Terms)

### Terminology note: “item universe” vs NLP usage
In recommender-systems literature, the fixed set of candidate items is commonly described as an **item item universe** or **product assortment**.
Because “topic” is overloaded in NLP, this project uses **concept/constraint** terminology for grounded field-value mentions.


Because “topic” is overloaded in NLP, this project uses **item universe-grounded** definitions:

- **Item**: a movie title in the item universe, \(m \in \mathcal{M}\).
- **Field / Facet**: an attribute type, \(f \in \mathcal{F}\), e.g., `genre`, `actor`, `director`, `writer`, `language`, `year`, `name`, `plot`.  
  (`plot` values are controlled plot keywords extracted from item universe summaries.)
- **Field value**: a concrete value, \(v \in \mathcal{V}_f\), e.g., “Thriller”, “Dev Patel”.
- **Item universe concept**: a grounded pair \(c = (f, v)\). In code, represented as the string `"f=v"`.
- **Item universe concept set for utterance \(x\)**:  
  \[
  E(x) \subseteq \mathcal{C}
  \]
  extracted by matching utterance text against item universe values (canonicalized).

**Legacy naming note:** `TOPIC_FIELDS` is a backward-compatible alias of `CONCEPT_FIELDS`.

- **Preference focus**: which field(s) dominate the user’s constraints at a time (genre-centric, actor-centric, etc.).  
  *(This can be implemented explicitly in the simulator as `current_focus_field`.)*

> **Do not claim** you do LDA topic modeling. This is **constraint tracking / concept grounding**.

---

## 3) Global Data Contracts (Schemas) — MUST NOT BREAK

### 3.1 Conversation (in-memory)
A conversation is a list of dicts:

- USER turn:
```json
{"speaker":"USER","text":"...","user_meta":{...}}
```

- SYSTEM turn:
```json
{"speaker":"SYSTEM","text":"...","constraints":{...}}
```

### 3.2 Log file (saved JSON)
Saved by `simulate.py` to `logs/<model>_session_<id>.json`:

```json
{
  "session_id": "<id>",
  "model_name": "<model>",
  "true_genre": "<user.true_genre>",
  "conversation": [ ...turn dicts... ],
  "shift_events": [ ... ]
}
```

### 3.3 Results file (`results.jsonl`)
Each line written by `batch_run.py`:

```json
{
  "model": "<model>",
  "...": "metrics from evaluate.evaluate(...)",
  "judge": { "...": "from evaluate.llm_judge(...)" }
}
```

> **If you rename fields**, update **all** of: `batch_run.py`, `analyze_results.py`, and any plotting/CSV outputs.

---

## 4) End-to-End Pipeline (Input → Processing → Output)

### Stage A — Item universe world construction (`dataset.py`)
**Input:** movie JSON corpus.  
**Outputs (in-memory):**
- `MOVIE_DB`: title → metadata dict
- `ATTRIBUTE_INDEX[field][value] → [titles...]` for fast constraint lookup
- `ATTRIBUTE_CANONICAL` + `ATTRIBUTE_VALUES` for normalization and grounding
- `PLOT_KEYWORD_INDEX` (if enabled) for controlled plot descriptors

### Stage B — Dialogue simulation (`simulate.py`)
For `N_TURNS`:
1. `UserSimulator.get_message()` generates USER utterance (may shift preference).
   `simulate.py` attaches `user_meta` via `UserSimulator.get_last_meta()`.
2. `CRSSystem.respond(user_msg)`:
   - updates internal `constraints`,
   - calls `dataset.recommend(constraints)`,
   - prompts the LLM with top item universe candidates,
   - returns SYSTEM natural-language response.
3. Append USER + SYSTEM turns, store `user_meta` + constraints snapshots.
4. Save a log JSON file (includes `shift_events` for debugging).

### Stage C — Quantitative evaluation (`evaluate.py` + `feature_extraction.py`)
Given a conversation:
1. Extract grounded concept sets \(E(U_t), E(S_t)\) for each turn.
2. Compute **turn-level** scores:
   - **CC(t)**: Jaccard overlap of concept sets.
   - **CR(t)**: cosine similarity of IDF-weighted concept vectors.
   - **I(t)**: n-gram copy ratio (penalty).
3. Aggregate over turns → **Turn Adaptation Score (TAS)**.
4. Detect user segments (topic boundaries) and compute recovery metrics:
   - **grounded**: use simulator `shift_event` metadata
   - **legacy**: use embedding/Jaccard thresholding
5. Aggregate → **Conversation Adaptation Score (CAS)**.

### Stage D — LLM-as-judge (optional)
`llm_judge(conversation)` prompts a rubric for holistic scores (coherence/personalization/proactiveness).  
Use for **convergent validity** analysis, not as truth.

### Stage E — Batch experiments (`batch_run.py`)
- Runs multiple sessions for each CRS model.
- Writes `results.jsonl`, plus summary CSVs.

### Stage F — Analysis and visualization (`analyze_results.py`)
- Loads results, normalizes judge scores, generates plots (distributions, radar, correlation scatter, etc.)

---

## 5) Formal Metric Definitions (Grounded + Interpretable)

### 5.1 Concept extraction (grounding)
For utterance \(x\), extractor:
\[
E(x) = \{ (f,v) \in \mathcal{C} : v \text{ is matched in } x \text{ under field } f \}
\]
In code, \(E(x)\) becomes a set of strings: `{"actor=dev patel", "genre=thriller", ...}`.

> **Limitation:** grounding is only as good as normalization and alias lists; paraphrases may be missed.

### 5.2 Cross-Coherence / Concept Overlap (CC)
Turn-level:
\[
CC(t) = \frac{|E(U_t)\cap E(S_t)|}{|E(U_t)\cup E(S_t)|}
\]
Interprets discrete agreement in grounded item universe concepts.

### 5.3 Context Retention / Constraint Similarity (CR)
Define an IDF-weighted vector \( \mathbf{a}(x)\) over the item universe concept vocabulary.
Cosine similarity:
\[
CR(t)=\cos(\mathbf{a}(U_t),\mathbf{a}(S_t))=
\frac{\mathbf{a}(U_t)\cdot \mathbf{a}(S_t)}{\|\mathbf{a}(U_t)\|\ \|\mathbf{a}(S_t)\|}
\]

Why cosine here:
- Handles **weighted emphasis** (IDF reduces dominance of ubiquitous concepts).
- Works with accumulating constraints (vector space lets partial matches contribute).

### 5.4 Topic Interference / Copy Penalty (I)
Using word n-grams of orders \(n\in \mathcal{N}\):
\[
I(t)=\max_{n\in\mathcal{N}}
\frac{|G_n(S_t)\cap G_n(U_t)|}{|G_n(S_t)|+\epsilon}
\]
Penalizes degenerate “echoing” behavior.

### 5.5 Topic Adaptation Score (TAS)
Aggregate over turns:
\[
\overline{CC}=\frac{1}{T}\sum_{t=1}^T CC(t),\quad
\overline{CR}=\frac{1}{T}\sum_{t=1}^T CR(t),\quad
\overline{I}=\frac{1}{T}\sum_{t=1}^T I(t)
\]
\[
TAS = \alpha \overline{CC} + \beta \overline{CR} - \gamma \overline{I}
\]
Default weights are neutral; weights can be tuned or sensitivity-tested.
**Implementation note:** the current code clips per-turn TAS to \([-1, 1]\) for stability.

### 5.6 Segment detection + recovery (CAS components)
Define **user segments** as contiguous turn ranges where user concept distribution is stable.  
Segments are defined by simulator `shift_event` metadata in grounded mode; legacy mode can
use embedding similarity and/or topic-set Jaccard drops.

Recovery metrics (dialogue-level):
- **Topic recovery rate**: fraction of shifts after which system realigns within window \(W\).
- **Average recovery delay**: mean number of turns to recover alignment after a shift.
- **Topic interference**: copying/contamination around shifts.

CAS is a weighted combination in `CAS_WEIGHTS`.

> **Design principle:** TAS is local/turn-level; recovery metrics are shift-aware diagnostics.

### 5.7 Concept delta diagnostics (missing vs hallucinated)
For each turn (and each segment), compute:
- **Missing concepts**: \(E(U_t) \setminus E(S_t)\)
- **Hallucinated concepts**: \(E(S_t) \setminus E(U_t)\)

These are surfaced as per-turn lists plus aggregate totals and top-k frequent concepts.

---

## 6) Module Responsibilities (File-by-File Truth)

### `config.py`
- Central constants: mode, models, thresholds, weights, field lists, paths.
- **Single source of truth** for: `TAS_WEIGHTS`, `CAS_WEIGHTS`, `RECOVERY_WINDOW`,
  `ALIGNMENT_THRESHOLD`, and `TOPIC_EXTRACTOR_MODE`.

### `dataset.py`
- Loads item universe, canonicalizes values, builds indexes.
- Implements:
  - `recommend(constraints)` (intersection/union retrieval)
  - `get_attribute_values(field)` (attribute pool for simulator)
  - `find_attribute_in_text(field, text)` (lexical grounding helper)
  - `find_plot_keywords_in_text(text)` (plot keyword grounding)
- Optional plot keyword indexing for controlled descriptors.

### `feature_extraction.py`
- Defines the grounded extractor and vectorization:
  - `extract_fielded_terms(text)`
  - `topic_set(text)` → for CC/Jaccard
  - `attribute_vector(text)` → for CR/cosine (IDF-weighted)
  - `copy_ratio(sys, user)` → for I (aliases: `copying_ratio`, `ngram_copy_ratio`)

### `system.py`
- CRS agent:
  - updates `self.constraints`
  - retrieves candidates via `dataset.recommend`
  - prompts LLM to recommend from item universe candidates
  - returns natural-language response
- **Note:** item universe enforcement is prompt-soft unless explicit post-processing is added.

### `user_sim.py`
- User simulator:
  - generates user utterances with controlled exploration and shifts
  - shift mechanism reuses overlapping field values (actor/director/...)
  - stores system history for continuity
- Optional improvement (recommended): explicit `current_focus_field` state machine.

### `simulate.py`
- Orchestrates a single dialogue run, saves a log file.
- Must preserve log schema.

### `evaluate.py`
- Computes:
  - TAS components and TAS
  - segment detection and recovery metrics
  - CAS
  - LLM judge scores (optional)
- Must preserve results schema consumed by `batch_run.py` and `analyze_results.py`.

### `batch_run.py`
- Runs multi-model × multi-session experiments.
- Writes `results.jsonl`, summary CSVs.

### `analyze_results.py`
- Loads results, flattens judge scores, produces plots.

---

## 7) Change Management Protocol (How to Modify Safely)

### 7.1 What must be updated together
If you change any of these, update all dependents:

- **Schemas** → update writers + readers + analysis scripts.
- **Concept fields** → update `feature_extraction.FeaturizerConfig.topic_fields`,
  grounding vocab, and any plots/labels.
- **Extraction rules** → update metric definitions and thesis text.

### 7.2 Required “sanity checks” after changes
After any nontrivial change, run:

1. **One simulation** → log file exists + schema matches.
2. **One evaluation** → TAS components non-NaN, values in expected ranges:
   - CC, CR in [0,1]; I in [0,1]; TAS in [-1, 1] (current implementation clamps).
3. **Batch run (small)** → results.jsonl lines parseable.
4. **Analyze** → plots generated without exceptions.

### 7.3 Version & dependency rules
- If adding dependencies (e.g., `scipy` for tests), either:
  - add to requirements and guard imports, OR
  - implement fallback computation in pure Python.

---

## 8) Conceptual Risk Register (Always Flag If Triggered)

1. **Grounding false negatives**: paraphrases not matched → CC/CR underestimated.  
   Mitigation: alias tables, normalization improvements, optional embeddings for plot keywords.

2. **Soft item universe constraint**: system may recommend non-item universe items.  
   Mitigation: add post-processing enforcement or structured output (JSON) constraints.

3. **Simulator bias**: user shift policy may be narrow.  
   Mitigation: implement explicit focus-type shifts + report simulator policy.

4. **Turn-only alignment limitations**: delayed adaptation missed.  
   Mitigation: segment metrics + recovery delay (already planned).

5. **LLM-as-judge variability**: prompt/model sensitivity.  
   Mitigation: report judge prompt, use correlation/CI, do not treat as truth.

---

## 9) Statistical Testing + Validity Checks (Recommended)

### 9.1 Convergent validity with judge
Compute rank correlation between TAS/CAS and judge scores:
- Spearman’s \(\rho\), optionally Kendall’s \(\tau\).
- Bootstrap CI for correlation.
- Scatter plot TAS vs judge with \(\rho\).

### 9.2 Construct validity using controlled shifts
Because shifts are scripted/detected:
- CC should drop at a shift and recover if system adapts.
- CR should be higher within stable segments than across segments.

Tests:
- Paired comparisons (Wilcoxon signed-rank or paired t-test) across sessions.

### 9.3 Ablations
Report:
- TAS without copy penalty
- CC-only, CR-only
- CAS without recovery terms (if CAS exists)

---

## 10) Visualization Checklist (for Thesis)

- Per-turn timeline: CC(t), CR(t), I(t) with shift markers.
- Box/violin plots of TAS components per model.
- Radar chart of component means per model.
- Scatter TAS vs Judge with Spearman \(\rho\) and CI.
- Recovery delay histogram + recovery rate vs window \(W\).
- Ablation bars comparing variants.

---

## 11) What to Update When Something Changes (Quick Table)

| Change | Must Update |
|---|---|
| Add new field (e.g., `plot`) | `feature_extraction.FeaturizerConfig.topic_fields`, `dataset` indexing, `feature_extraction` vocab, thesis definitions |
| Change extraction (regex/normalization) | `feature_extraction`, evaluation validity notes, risk register |
| Change log/result schema | `simulate`, `batch_run`, `analyze_results`, this doc |
| Add new metrics | `evaluate`, `batch_run` output columns, `analyze_results` plots, this doc |
| Change simulator policy | `user_sim`, construct-validity argument, this doc |

---

## 12) File Inventory (Expected)
- `config.py`
- `dataset.py`
- `feature_extraction.py`
- `system.py`
- `user_sim.py`
- `simulate.py`
- `evaluate.py`
- `batch_run.py`
- `analyze_results.py`
- `logs/` (generated)
- `results/` or `results.jsonl` (generated)
- `figures/` (generated)

---

## 13) Final Guardrail: What This System Claims (and Does Not Claim)

✅ Claims:
- Evaluates **prompted CRS adaptation** in a **closed item universe** using **grounded concept overlap** and **weighted constraint similarity**, plus **copying penalties** and **shift recovery diagnostics**.

❌ Does not claim:
- Supervised recommendation accuracy, retrieval metrics like Recall@K.
- True semantic topic modeling (LDA).
- Human-ground-truth evaluation (unless you add it).

---

**Maintainer note:**  
If any part of the code starts to contradict this document, the code is the source of truth for behavior—but the document must be updated immediately, and thesis claims must follow the updated definitions.
