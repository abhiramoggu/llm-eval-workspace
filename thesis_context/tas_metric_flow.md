# Topic Adaptation Score (TAS) / Context Adaptation Score (CAS) Flow

This project computes a Context Adaptation Score (CAS) in code. In the thesis, you can present it as the Topic Adaptation Score (TAS) with the same formulation.

## Notation
- u_i: user utterance at turn i
- s_i: system response at turn i
- sim(a,b): cosine similarity between embeddings of a and b (sentence-transformer if enabled, otherwise TF-IDF fallback)
- topics(x): extracted topic keywords from text x
- jaccard(A,B) = |A intersect B| / |A union B|

## Topic extraction
- Mode controlled by `TOPIC_EXTRACTOR_MODE` in `context-aware/config.py`.
  - lda: use pre-trained LDA model on movie plots (trained by `context-aware/train_lda.py`).
  - llm: prompt an LLM to return a JSON list of topics.
  - heuristic: keyword scoring with domain hints and stopwords.
- Output: up to 5 topic strings per utterance.

## User segment detection (topic shifts)
1. For each user utterance, compute:
   - sim(prev_text, current_text)
   - jaccard(prev_topics, current_topics)
2. Declare a shift if:
   - sim < SIM_TOPIC_SHIFT, OR
   - jaccard < TOPIC_JACCARD_SHIFT
3. Create contiguous user segments with aggregated topics.

## Core metrics (computed in `context-aware/evaluate.py`)

### 1) Cross-coherence
Measures how well each system response aligns with the immediately preceding user message.

Formula:
  cross_coherence = (1/N) * sum_{i=1..N} sim(u_i, s_i)

### 2) Context retention
Measures how consistent consecutive system responses are across the dialogue.

Formula:
  context_retention = (1/(M-1)) * sum_{i=2..M} sim(s_{i-1}, s_i)

### 3) Topic recovery rate and recovery delay
For each detected user topic shift (from old topics to new topics):
- Scan system turns after the shift.
- Compute alignment between system topics and new topics.
- Recovery occurs when alignment >= ALIGNMENT_THRESHOLD.

Per-shift recovery delay = number of system turns until recovery.

Formulas:
  topic_recovery_rate = recovered_shifts / total_shifts
  avg_recovery_delay = mean(recovery_delay over recovered shifts)

Alignment and leakage details:
- alignment = jaccard(sys_topics, new_topics)
  - If jaccard < 0.3, fall back to sim(" ".join(sys_topics), " ".join(new_topics)).

### 4) Topic interference
Penalizes old-topic leakage after a new topic shift.

For each shift:
- leak = jaccard(sys_topics, old_topics)
  - If jaccard < 0.3, fall back to sim(" ".join(sys_topics), " ".join(old_topics)).
- A leakage hit occurs if leak >= ALIGNMENT_THRESHOLD.
- interference_per_shift = leakage_hits / denom,
  where denom is either recovery delay (if recovered) or min(sys_seen, 4).

Overall:
  topic_interference = mean(interference_per_shift)

## TAS/CAS aggregation
The composite score is a weighted average of the above metrics, with normalization:
- avg_recovery_delay normalized to [0,1] with range 1..6, then inverted.
- topic_interference normalized to [0,1] and inverted.

Let:
  rd_norm = 1 - clamp((avg_recovery_delay - 1)/(6 - 1))
  ti_norm = 1 - clamp(topic_interference)

Then:
  TAS = sum_k (w_k * v_k) / sum_k (w_k)

where v_k in {topic_recovery_rate, rd_norm, ti_norm, cross_coherence, context_retention}.

Weights are defined in `context-aware/config.py` under `CAS_WEIGHTS`.

## Why these components
- Cross-coherence captures immediate response alignment.
- Context retention captures stability across system turns.
- Recovery rate and delay capture speed and frequency of adaptation.
- Topic interference captures whether old preferences leak after a shift.

## Notes for the paper
- The implementation uses embedding similarity between full utterances for cross-coherence and retention, not only topic words.
- Jaccard similarity is primarily used for topic overlap in shift detection and recovery checks.
- Make the chosen topic extractor mode and embedding backend explicit in the methodology.
