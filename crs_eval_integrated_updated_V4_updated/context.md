# CRS Preference-Shift Evaluation — Developer Context (Living Document)

> **Purpose of this file**
> - This document is the **single source of truth** for how the codebase implements the thesis pipeline.
> - It is designed to be pasted into an LLM coding agent (Codex) so the agent can **modify code safely** without breaking assumptions.
> - It is a **living document**: whenever you change prompts, pipeline logic, metrics, or data schemas, you must update the relevant sections here.

---

## 0) What the repository does (one paragraph)

The repository simulates multi-turn dialogues between two agents—(1) a fixed-item-set-constrained conversational recommender system (CRS) and (2) a user simulator that introduces controlled preference shifts—and then evaluates how well the CRS adapts over time. The evaluation is grounded in a closed-domain movie item set (fixed candidate universe) so that fielded concept/constraint evidence can be extracted directly from known metadata rather than guessed from open-ended text. The main proposed metric is **Adaptation Score (TAS components)**, which combines (i) **Cross-Coherence** (Jaccard overlap of catalog-grounded topics), (ii) **Context Retention** (cosine similarity between catalog-grounded attribute vectors), and (iii) **Topic Interference** (word n-gram copying penalty). An optional judge ("PEPPER"-style LLM-as-a-Judge) can produce holistic conversation quality scores, but TAS is intended to provide a more **grounded, diagnostic alternative**.

**Terminology note:** Some code still uses the word *topic* as a legacy label.
In the thesis/docs, *topic* means **catalog-grounded concept/constraint set** (field=value), not LDA topics.
`TOPIC_FIELDS` is a legacy alias for `CONCEPT_FIELDS`.

---

## 1) Files and their responsibilities (map)

### 1.1 Configuration
- **`config.py`**: Central configuration (modes, models, evaluation thresholds, weights).
  - Important constants used by multiple modules:
    - `TAS_WEIGHTS`: alpha/beta/gamma used in TAS = alpha·CC + beta·CR − gamma·I.
    - `RECOVERY_WINDOW`, `ALIGNMENT_THRESHOLD`, `EVAL_MODE`, `TOPIC_EXTRACTOR_MODE`.
  - Grounded concept fields live in `feature_extraction.FeaturizerConfig.topic_fields`.

### 1.2 Data and catalog indexing
- **`dataset.py`**: Loads and indexes the movie catalog.
  - Holds the `MOVIE_DB` dict (`title -> movie_metadata`).
  - Builds lookup indices for fast retrieval of movies by attribute values.
  - Exposes:
    - `recommend(constraints)`: returns catalog items matching constraints.
    - `get_available_genres()`: returns normalized genres with sufficient coverage.
    - `get_attribute_values(field)`: returns global pool of values for a field.
    - `find_attribute_in_text(field, text)`: lexical grounding helper.
    - `find_plot_keywords_in_text(text)`: plot keyword grounding helper.

### 1.3 Simulation (conversation generation)
- **`user_sim.py`**: User simulator policy.
  - Maintains user *preference focus* (e.g., genre/actor/director) for the conversation anchor, but can request constraints across multiple fields.
  - **Updated behavior**: topic shifts are generated coherently using anchor metadata (bridge concepts) and catalog-backed sampling.
- **`system.py`**: CRS agent wrapper.
  - Parses user utterances for constraints.
  - Retrieves candidate items using `dataset.recommend`.
  - Produces a natural-language response (mock or via Ollama).
  - Ensures any recommended titles exist in the catalog.
- **`simulate.py`**: Orchestrates a single run:
  - Alternates USER/SYSTEM turns.
  - Logs conversation to disk (JSON) along with model name, `session_id`, and `shift_events`.
- **`batch_run.py`**: Runs many simulations (multiple models, multiple seeds) and writes results.

### 1.4 Feature extraction and evaluation
- **`feature_extraction.py`**: **New** catalog-grounded featurizer.
  - Builds restricted vocabulary from:
    - Structured fields (TAS/diagnostics): `genre, actor, director, writer, language, year`
    - Optional exploratory plot keywords (`plot_kw`), disabled by default for TAS
  - Provides:
    - `topic_set(text)`: returns a set of fielded tokens like `actor=dev patel`.
    - `attribute_vector(text, active_fields=...)`: returns a dense vector over the restricted vocabulary.
    - `copy_ratio(sys_text, user_text, n_orders)`: the copying penalty I.
- **`evaluate.py`**: Computes metrics for a conversation.
  - Computes TAS components per USER→SYSTEM pair:
    - `cross_coherence`: average Jaccard of topic sets.
    - `context_retention`: average cosine similarity of attribute vectors.
    - `topic_interference`: average n-gram copy ratio.
  - Computes:
    - `topic_adaptation_score` (TAS)
    - Keeps a legacy “CAS” composite (optional) for comparison.
  - Optionally calls a judge model to produce holistic rubric scores.

### 1.5 Results analysis
- **`analyze_results.py`**: Loads JSONL results and plots distributions and correlations.
  - Updated to recognize both:
    - `topic_interference` (copy penalty used by TAS)
    - `topic_interference_leakage` (legacy recovery diagnostic)

---

## 2) End-to-end pipeline (step-by-step)

### Step A — Load and index the catalog (`dataset.py`)
1. Load the movie DB into `MOVIE_DB`.
2. Normalize attribute strings to reduce duplicates.
3. Build inverted indices per field so we can quickly retrieve candidates.

### Step B — Simulate a dialogue (`simulate.py`)
1. Instantiate:
   - `UserSimulator` (USER)
   - CRS system wrapper (SYSTEM)
2. For each turn t:
   - USER produces an utterance using:
     - initial request → refinement → coherent shift (via overlap) → refinement → ...
   - SYSTEM parses constraints and returns recommendations/questions.
3. The run outputs a conversation list:
   - `[{'speaker':'USER','text':...,'user_meta':...}, {'speaker':'SYSTEM','text':...,'constraints':...}, ...]`

### Step C — Extract grounded signals (`feature_extraction.py`)
For each USER/SYSTEM pair at turn t:
1. `topics(USER_t)`: fielded set (e.g., `genre=thriller`, `actor=dev patel`).
2. `topics(SYSTEM_t)`: same extraction.
3. `attrs(USER_t)`: attribute vector restricted to catalog vocabulary.
4. `attrs(SYSTEM_t)`: same.
5. `copy_ratio(SYSTEM_t, USER_t)`: n-gram overlap penalty.

### Step D — Compute TAS (`evaluate.py`)
For each USER→SYSTEM pair t:
- **CC(t)** = Jaccard(topics(USER_t), topics(SYSTEM_t))
- **CR(t)** = cosine(attrs(USER_t), attrs(SYSTEM_t))
- **I(t)**  = copy_ratio(SYSTEM_t, USER_t)

Aggregate over turns:
- CC = mean_t CC(t)
- CR = mean_t CR(t)
- I  = mean_t I(t)

Final:
- **TAS = alpha·CC + beta·CR − gamma·I**

---

## 3) Key design choices (why this is defensible)

### 3.1 Grounding: why restrict to catalog vocabulary?
- Unrestricted TF–IDF or embeddings can overweight irrelevant words ("good", "movie", "like").
- Using only structured fields and a controlled plot keyword list:
  - reduces noise,
  - improves interpretability,
  - makes the metric *auditable* (you can inspect what matched).

### 3.2 Why Jaccard for Cross-Coherence (CC)?
- Topics are treated as **sets of discrete catalog facts** (fielded tokens).
- Jaccard measures overlap regardless of ordering or surface phrasing.

### 3.3 Why cosine for Context Retention (CR)?
- Attribute vectors represent soft alignment across multiple constraints.
- Cosine normalizes for verbosity and measures orientation/similarity.

### 3.4 Why n-gram copying penalty for Interference (I)?
- Detects “echoing” failure mode: the CRS repeats the user instead of reasoning.
- Works even when topic overlap is high but dialogue quality is low.

---

## 4) What to change when you extend the system

### 4.1 Adding new topic fields
If the dataset adds fields (e.g., `country`, `runtime`):
1. Update `feature_extraction.FeaturizerConfig.topic_fields`.
2. Update `dataset.py` normalization/indexing for the new field.
3. Update `feature_extraction.py` vocabulary builder to include the new field.
4. Update user simulator to use the field in refinements and/or shift bridges.

### 4.2 Changing TAS weights or components
1. Update `config.TAS_WEIGHTS`.
2. Update `evaluate.compute_tas` if you add new components.
3. Update `analyze_results.py` to plot/correlate the new columns.

### 4.3 Updating prompts (CRS / UserSim / Judge)
Maintain these invariants:
- CRS must recommend **only** items in the catalog.
- UserSim shifts must be coherent and controlled.
- Judge rubric should be kept **separate** from grounded metrics.

---

## 5) Codex / LLM coding agent instructions

When editing this repository:
1. **Do not** change result JSON keys unless you also update analysis scripts.
2. Preserve the alternating turn format in conversation logs.
3. Keep `topic_adaptation_score` as the main output metric.
4. If you add or remove a metric, update:
   - `evaluate.py` output keys
   - `analyze_results.py` column expectations
5. After modifications:
   - run a small simulation (e.g., 2 models × 2 runs)
   - confirm the JSONL outputs include TAS and components

---

## 6) Change log (fill this in whenever you modify the pipeline)

- **YYYY-MM-DD**: <what changed> / <why> / <files touched>
