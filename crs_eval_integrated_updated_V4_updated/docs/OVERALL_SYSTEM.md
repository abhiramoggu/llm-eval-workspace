# CRS Preference-Shift Evaluation — System Overview (Living Spec)

This repository implements an evaluation framework for **LLM-based conversational recommender systems (CRS)** under a **closed movie catalog**.

The core idea is to simulate multi-turn conversations where a **User Simulator** issues **controlled preference-focus shifts** (e.g., genre → actor → director) while a **CRS Agent** responds with catalog-constrained recommendations. After the conversation, the evaluation computes an interpretable metric **Topic Adaptation Score (TAS)** and diagnostics such as **recovery rate**.

---

## 1. Terminology (thesis-safe)

### World / Catalog level
- **Item**: a movie title, denoted \(m \in \mathcal{M}\).
- **Field / Facet**: an attribute type, denoted \(f \in \mathcal{F}\).  
  Structured fields used for TAS: `genre`, `actor`, `director`, `writer`, `language`, `year`.  
  `name` (titles) and `plot_kw` are excluded from TAS by default and only used for appendix/debug.
- **Field value**: a concrete value \(v \in \mathcal{V}_f\).  
  Example: (`actor`, `Dev Patel`).

### Utterance level
- An **utterance** is natural language text produced by either the USER or SYSTEM.

### Catalog-grounded concept
- A **catalog concept** is a grounded pair \(c = (f, v)\) where \(v\) matches a value in the catalog.  
  Plot keywords (`plot_kw`) are optional exploratory descriptors and are not part of TAS unless enabled.

**Legacy naming note:** Some code still uses “topic” as a legacy label. In this repository,
`TOPIC_FIELDS` is a backward-compatible alias of `CONCEPT_FIELDS`, and “topic” refers to
catalog-grounded concepts (not LDA topics).

### Extractor
- \(E(x) \subseteq \mathcal{C}\) extracts the set of catalog concepts mentioned in utterance \(x\).

### Preference focus
- **Preference focus** is the field currently steering the user's intent (e.g., actor-centric vs genre-centric).  
  In code: `user_meta.focus_field` + `user_meta.focus_value`.

### Shift event
- A **shift event** occurs when the simulator changes preference focus using an overlapping **bridge concept** from the current context.  
  In code: `user_meta.shift_event=True`.

---

## 2. Pipeline (end-to-end)

1. **Load catalog**  
   `dataset.py` loads JSON, normalizes fields, builds indices.

2. **Simulate dialogue**  
   `simulate.py` alternates:
   - USER: `UserSimulator.get_message()`  
     → `simulate.py` attaches `user_meta` via `UserSimulator.get_last_meta()` (focus, constraints, shift_event).
   - SYSTEM: `CRSSystem.respond()`  
     → constrained to catalog via `dataset.recommend()`.

3. **Evaluate**  
   `evaluate.py` computes:
   - Turn-level alignment (USER_t vs SYSTEM_t)
   - Segment-level alignment (aggregate within simulator-defined segments)
   - Shift recovery metrics (recover within window after shift)

4. **Aggregate across models**  
   `batch_run.py` runs multiple models over paired simulator seeds, writes `results.jsonl`.

5. **Statistical tests + plots**  
   `analyze_results.py` computes:
   - Correlations (Spearman + bootstrap CI)
   - Paired tests (Wilcoxon/paired t) across models
   - Visualizations (scatter, timelines, distributions)

---

## 3. TAS (grounded metric)

For each turn \(t\), let USER utterance be \(U_t\), SYSTEM utterance be \(S_t\).

### Cross-Coherence (CC)
Discrete overlap of grounded concepts:

\[
CC(t) = \frac{|E(U_t) \cap E(S_t)|}{|E(U_t) \cup E(S_t)|}
\]

### Context Retention (CR)
Weighted similarity of grounded constraints using TF–IDF-like vectors \(\phi(x)\):

\[
CR(t) = \frac{\phi(U_t) \cdot \phi(S_t)}{\|\phi(U_t)\|\,\|\phi(S_t)\|}
\]

### Copying Penalty (I)
Surface overlap penalty using word n-grams:

\[
I(t) = \max_{n \in \{3,4\}} \frac{|G_n(S_t) \cap G_n(U_t)|}{|G_n(S_t)|}
\]

### TAS
\[
TAS(t) = \alpha\,CC(t) + \beta\,CR(t) - \gamma\,I(t)
\]
and dialogue-level TAS is typically the mean over turns.

Weights are in `config.TAS_WEIGHTS`.

---

## 4. Shift recovery diagnostics

Let \(\mathcal{S}\) be the set of shift turns (from simulator metadata).  
A shift at turn \(s\) is considered **recovered** if there exists \(d \in [0, W-1]\) such that:

\[
CC(s+d) \ge \tau
\]

Where:
- \(W\) = `RECOVERY_WINDOW`
- \(\tau\) = `ALIGNMENT_THRESHOLD`

Metrics:
- **Recovery rate** = recovered shifts / total shifts
- **Average recovery delay** = mean smallest \(d\) among recovered shifts

---

## 5. What “segment-level” means here

Segments are defined by simulator shift points (ground truth).  
For a segment spanning turns \([a,b]\), we aggregate concepts/vectors across turns, then compute CC/CR on the aggregated representations.

---

## 6. What must be stated as limitations (thesis-safe)

- **Extractor validity**: catalog grounding is primarily string/lexicon based → paraphrases can be missed.
- **Simulator bias**: shift policy constrains the space of preference transitions; must be reported.
- **LLM-as-CRS nondeterminism**: prompts may yield different outputs; use multiple sessions.
- **LLM-as-judge unreliability**: use only as convergent evidence (correlation), not ground truth.

---

## 7. How to keep this “living spec” updated

Whenever you modify:
- simulator policy,
- extractor fields/vocabulary,
- TAS definition,
- evaluation window/threshold,
- new plots/tests,

update:
- this file (`docs/OVERALL_SYSTEM.md`),
- the corresponding per-file markdown in `docs/`,
- and the top-level `context.md` used for Codex.
