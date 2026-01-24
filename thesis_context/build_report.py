import csv
import json
import os
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.graphics import renderPM


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_METRICS_PATH = os.path.join(ROOT_DIR, "context-aware", "model_metrics.csv")
RESULTS_PATH = os.path.join(ROOT_DIR, "context-aware", "results.jsonl")

MD_PATH = os.path.join(BASE_DIR, "technical_writeup.md")
PDF_PATH = os.path.join(BASE_DIR, "technical_writeup.pdf")


def read_model_metrics(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def coerce_metrics(rows, metric_cols):
    for row in rows:
        for m in metric_cols:
            if m in row and row[m] not in (None, ""):
                row[m] = float(row[m])
    return rows


def top_two_by_metric(rows, metric, lower_is_better=False):
    data = [r for r in rows if metric in r]
    data = sorted(data, key=lambda r: r[metric], reverse=not lower_is_better)
    return data[:2]


def read_results_counts(path):
    counts = {}
    shifts = {}
    total_shifts = 0
    total_sessions = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            model = rec.get("model", "unknown")
            counts[model] = counts.get(model, 0) + 1
            shifts.setdefault(model, []).append(rec.get("num_topic_shifts", 0))
            total_shifts += rec.get("num_topic_shifts", 0)
            total_sessions += 1
    avg_shifts = 0.0
    if total_sessions:
        avg_shifts = total_shifts / total_sessions
    return counts, shifts, avg_shifts


def draw_multiline_label(drawing, x_center, y_center, lines, font_size=7, leading=9):
    start_y = y_center + (len(lines) - 1) * leading / 2.0
    for idx, line in enumerate(lines):
        y = start_y - idx * leading
        drawing.add(
            String(
                x_center,
                y,
                line,
                textAnchor="middle",
                fontName="Helvetica",
                fontSize=font_size,
            )
        )


def add_box(drawing, x, y, w, h, lines, fill_color=colors.whitesmoke, stroke_color=colors.black):
    drawing.add(Rect(x, y, w, h, fillColor=fill_color, strokeColor=stroke_color))
    draw_multiline_label(drawing, x + w / 2.0, y + h / 2.0, lines)


def add_arrow(drawing, x1, y1, x2, y2, size=4):
    drawing.add(Line(x1, y1, x2, y2))
    # Only horizontal right-facing arrows are used here.
    drawing.add(
        Polygon(
            [x2, y2, x2 - size, y2 - size, x2 - size, y2 + size],
            fillColor=colors.black,
            strokeColor=colors.black,
        )
    )


def build_system_overview_diagram(path):
    width, height = 520, 170
    drawing = Drawing(width, height)
    box_w, box_h, gap = 90, 40, 10
    y = 110
    x = 10
    labels = [
        ["User", "Simulator"],
        ["CRS", "System"],
        ["Conversation", "Logs"],
        ["Evaluation", "TAS/CAS +", "LLM Judge"],
        ["Aggregate", "Metrics +", "Figures"],
    ]

    positions = []
    for lines in labels:
        add_box(drawing, x, y, box_w, box_h, lines)
        positions.append((x, y))
        x += box_w + gap

    for i in range(len(positions) - 1):
        x1 = positions[i][0] + box_w
        y1 = positions[i][1] + box_h / 2.0
        x2 = positions[i + 1][0]
        y2 = positions[i + 1][1] + box_h / 2.0
        add_arrow(drawing, x1, y1, x2, y2)

    # Movie KB box
    kb_x, kb_y, kb_w, kb_h = 10, 25, 180, 30
    add_box(drawing, kb_x, kb_y, kb_w, kb_h, ["Movie KB", "(OpenDialKG JSON)"])
    # Connect KB to first two boxes
    drawing.add(Line(kb_x + kb_w / 2.0, kb_y + kb_h, 55, y))
    drawing.add(Line(kb_x + kb_w / 2.0, kb_y + kb_h, 155, y))

    renderPM.drawToFile(drawing, path, fmt="PNG")


def build_dataset_pipeline_diagram(path):
    width, height = 520, 140
    drawing = Drawing(width, height)
    box_w, box_h, gap = 120, 40, 10
    y = 60
    x = 10
    labels = [
        ["OpenDialKG", "Movie JSON"],
        ["Normalize +", "Index"],
        ["Attribute +", "Plot Indices"],
        ["Used by", "Recommender,", "User Sim, LDA"],
    ]
    positions = []
    for lines in labels:
        add_box(drawing, x, y, box_w, box_h, lines)
        positions.append((x, y))
        x += box_w + gap

    for i in range(len(positions) - 1):
        x1 = positions[i][0] + box_w
        y1 = positions[i][1] + box_h / 2.0
        x2 = positions[i + 1][0]
        y2 = positions[i + 1][1] + box_h / 2.0
        add_arrow(drawing, x1, y1, x2, y2)

    renderPM.drawToFile(drawing, path, fmt="PNG")


def build_experimental_design_diagram(path):
    width, height = 520, 140
    drawing = Drawing(width, height)
    box_w, box_h, gap = 95, 40, 10
    y = 60
    x = 10
    labels = [
        ["6 CRS", "Models"],
        ["1000", "Sessions", "per Model"],
        ["20", "Turns", "per Session"],
        ["Evaluate", "TAS/CAS +", "PEPPER"],
        ["Aggregate", "+ Stats"],
    ]
    positions = []
    for lines in labels:
        add_box(drawing, x, y, box_w, box_h, lines)
        positions.append((x, y))
        x += box_w + gap

    for i in range(len(positions) - 1):
        x1 = positions[i][0] + box_w
        y1 = positions[i][1] + box_h / 2.0
        x2 = positions[i + 1][0]
        y2 = positions[i + 1][1] + box_h / 2.0
        add_arrow(drawing, x1, y1, x2, y2)

    renderPM.drawToFile(drawing, path, fmt="PNG")


def ensure_diagrams():
    system_path = os.path.join(BASE_DIR, "fig_system_overview.png")
    dataset_path = os.path.join(BASE_DIR, "fig_dataset_pipeline.png")
    experiment_path = os.path.join(BASE_DIR, "fig_experimental_design.png")

    if not os.path.exists(system_path):
        build_system_overview_diagram(system_path)
    if not os.path.exists(dataset_path):
        build_dataset_pipeline_diagram(dataset_path)
    if not os.path.exists(experiment_path):
        build_experimental_design_diagram(experiment_path)


def build_markdown(rows, counts, avg_shifts):
    metric_cols = [
        "topic_recovery_rate",
        "avg_recovery_delay",
        "topic_interference",
        "cross_coherence",
        "context_retention",
        "context_adaptation_score",
        "proactiveness",
        "coherence",
        "personalization",
    ]
    rows = coerce_metrics(rows, metric_cols)

    tops = {
        "topic_recovery_rate": top_two_by_metric(rows, "topic_recovery_rate"),
        "avg_recovery_delay": top_two_by_metric(rows, "avg_recovery_delay", lower_is_better=True),
        "topic_interference": top_two_by_metric(rows, "topic_interference", lower_is_better=True),
        "cross_coherence": top_two_by_metric(rows, "cross_coherence"),
        "context_retention": top_two_by_metric(rows, "context_retention"),
        "context_adaptation_score": top_two_by_metric(rows, "context_adaptation_score"),
        "proactiveness": top_two_by_metric(rows, "proactiveness"),
        "coherence": top_two_by_metric(rows, "coherence"),
        "personalization": top_two_by_metric(rows, "personalization"),
    }

    def fmt_top(metric_name, label, lower=False):
        items = tops[metric_name]
        if not items:
            return f"- {label}: not available"
        first = items[0]
        second = items[1] if len(items) > 1 else None
        if second:
            return (
                f"- {label}: {first['model']} ({first[metric_name]:.3f}) "
                f"then {second['model']} ({second[metric_name]:.3f})"
            )
        return f"- {label}: {first['model']} ({first[metric_name]:.3f})"

    sessions_note = ", ".join([f"{k}={v}" for k, v in sorted(counts.items())])

    md = f"""# Preference Shift Evaluation in Conversational Recommender Systems

## Proposed Approach
### System overview
This system evaluates how a conversational recommender system (CRS) adapts to user preference shifts over a multi-turn dialogue. A user simulator generates preference-seeking utterances with explicit shifts and contradictions. The CRS uses rule-based constraint extraction to retrieve candidate movies and then uses an LLM for response generation. The resulting conversation is evaluated by a topic-shift-aware metric (TAS/CAS) and a PEPPER-style LLM judge. Aggregate statistics and visualizations are produced across models.

![Figure 1. System overview of the evaluation pipeline.](fig_system_overview.png)

### Algorithm 1: End-to-end evaluation pipeline
```text
Input: movie KB, CRS model list, user simulator model, judge model
Output: per-session metrics, model summaries, figures

1. Load and index movie KB (genres, attributes, plot keywords).
2. For each CRS model m in the experiment list:
   2.1 For each session s in 1..N:
       a) Initialize UserSimulator and CRSSystem(m).
       b) For t in 1..20 turns:
          i) UserSimulator emits user message u_t.
         ii) CRSSystem updates constraints and returns response s_t.
        iii) Log (u_t, s_t) and update user simulator state.
       c) Evaluate conversation: segment user topics, compute TAS/CAS, run LLM judge.
       d) Write per-session result to results.jsonl.
3. Aggregate results across sessions and models into model_metrics.csv.
4. Generate plots and statistical tests.
```

### CRS component
The CRS uses rule-based extraction to update constraints from user utterances. Genre cues are matched against known genres, year is extracted with regex, and attribute values are matched against the dataset indexes (actor, director, language, writer, name). Plot keywords are extracted using a domain-specific keyword index. Based on the current constraints, candidate movies are retrieved from the dataset and supplied to the LLM as a short list. The LLM is only used to generate the final conversational response, not to parse constraints.

### User simulator
The simulator is a rule-based agent that generates preference-seeking utterances and explicit shifts. It rotates through asking about actors, themes, and mood, and periodically changes genre or attributes such as year, actor, director, language, writer, or plot. It can also emit contradictions to model realistic preference uncertainty. The simulator tracks system responses to choose follow-up questions and can optionally use an LLM to produce drifted utterances. This setup induces frequent topic shifts to stress-test adaptation.

### LLM scorer (PEPPER-style)
An LLM judge scores three qualitative metrics on a 1 to 5 scale: proactiveness, coherence, and personalization. The judge receives the full conversation and a short summary of user preferences extracted from user turns. These scores complement TAS/CAS by capturing conversational quality and perceived alignment.

### Topic Adaptation Score (TAS) / Context Adaptation Score (CAS)
The core metric is implemented as CAS in code, but it is presented as TAS in the thesis. TAS combines cross-coherence, context retention, recovery rate, recovery delay, and topic interference. Topic shifts are detected from user turns using embedding similarity and topic keyword overlap.

```text
cross_coherence = (1/N) * sum_i sim(u_i, s_i)
context_retention = (1/(M-1)) * sum_i sim(s_{{i-1}}, s_i)

Topic shift detection:
- shift if sim(prev_text, curr_text) < SIM_TOPIC_SHIFT
  or jaccard(prev_topics, curr_topics) < TOPIC_JACCARD_SHIFT

topic_recovery_rate = recovered_shifts / total_shifts
avg_recovery_delay = mean(recovery_delay over recovered shifts)

interference per shift = leakage_hits / denom
topic_interference = mean(interference per shift)

Normalization (range 1..6 for delay):
rd_norm = 1 - clamp((avg_recovery_delay - 1) / 5)
ti_norm = 1 - clamp(topic_interference)

TAS = sum_k (w_k * v_k) / sum_k (w_k)
where v_k in {{topic_recovery_rate, rd_norm, ti_norm, cross_coherence, context_retention}}
```

The implementation uses sentence-transformer embeddings when enabled; otherwise, it falls back to TF-IDF similarity. Topic extraction uses LDA by default, with LLM or heuristic fallback options.

## Dataset
The system uses an existing movie knowledge base derived from OpenDialKG in `opendialkg_movie_data.json`. No new dataset is created. Each movie includes fields such as name, genre, year, actor, director, language, writer, and plot.

![Figure 2. Dataset preprocessing and usage pipeline.](fig_dataset_pipeline.png)

Preprocessing steps include: (1) normalization of genre strings, (2) indexing of attribute values to enable fast constraint lookup, and (3) extraction of plot keywords for shallow semantic matching. The dataset supports three roles: recommendation retrieval in the CRS, attribute pools for the user simulator, and LDA training for topic extraction.

Fallbacks and checks: if pre-trained LDA files are missing, topic extraction falls back to a keyword heuristic. If transformer embeddings are unavailable, the metric uses TF-IDF vectors. Limitations include noisy genre labels and duplicate metadata entries, which can blur topic boundaries and reduce interpretability.

## Experiments
The experiments evaluate six CRS models with fixed user and judge settings. Each model is run for 1000 sessions and 20 turns per session. The session counts found in results.jsonl are: {sessions_note}. The average detected number of topic shifts per session is {avg_shifts:.2f}, indicating frequent user preference changes.

![Figure 3. Experimental design and evaluation flow.](fig_experimental_design.png)

Key settings:
- CRS models: gemma:2b, qwen:7b, qwen:4b, llama3:instruct, llama2:latest, mistral:7b.
- User simulator model: llama3:instruct.
- LLM judge model: llama3:instruct.
- Topic extractor mode: lda (default in config; can be llm or heuristic).
- Embedding backend: TF-IDF by default, sentence-transformer optional.
- Thresholds: SIM_TOPIC_SHIFT=0.55, TOPIC_JACCARD_SHIFT=0.35, ALIGNMENT_THRESHOLD=0.65.

Reported metrics:
- TAS/CAS components: topic_recovery_rate, avg_recovery_delay, topic_interference, cross_coherence, context_retention, and the weighted aggregate score.
- PEPPER-style metrics: proactiveness, coherence, personalization.
- Statistical testing: ANOVA with Tukey HSD post-hoc analysis.

## Results and Discussion
Overall, llama3:instruct leads the aggregate TAS/CAS score and all three PEPPER-style metrics, indicating strong alignment and conversational quality. Mistral:7b shows the best recovery rate, while qwen:7b shows the fastest average recovery. Gemma:2b has the lowest topic interference but weaker coherence and retention.

Top metrics from model_metrics.csv:
{fmt_top('context_adaptation_score', 'Best TAS/CAS')}
{fmt_top('topic_recovery_rate', 'Best recovery rate')}
{fmt_top('avg_recovery_delay', 'Fastest recovery (lower is better)')}
{fmt_top('topic_interference', 'Lowest interference (lower is better)')}
{fmt_top('cross_coherence', 'Highest cross-coherence')}
{fmt_top('context_retention', 'Highest context retention')}
{fmt_top('proactiveness', 'Best proactiveness (LLM judge)')}
{fmt_top('coherence', 'Best coherence (LLM judge)')}
{fmt_top('personalization', 'Best personalization (LLM judge)')}

![Figure 4. Average TAS/CAS by model.](../context-aware/figures/bar_charts/metric_context_adaptation_score_bar.png)

![Figure 5. LLM-as-judge radar (PEPPER-style).](../context-aware/figures/radars/llm_judge_radar.png)

![Figure 6. Correlation between quantitative and LLM-judge metrics.](../context-aware/figures/correlations/metrics_correlation_heatmap.png)

Tradeoffs are visible across models. Higher recovery rate does not guarantee higher cross-coherence or retention, and low interference alone does not yield the top TAS/CAS score. The correlation heatmap shows where subjective judgments track or diverge from automated adaptation metrics.

Threats to validity include: (1) dependence on topic extraction mode and thresholds, (2) the LLM judge using a single model and prompt, and (3) user simulator noise, including occasional unnatural title references due to dataset indexing. These factors should be reported alongside results to avoid over-claiming causal conclusions.
"""
    return textwrap.dedent(md)


def write_markdown(md_text):
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)


def scaled_image(path, max_width):
    img = ImageReader(path)
    iw, ih = img.getSize()
    scale = min(max_width / iw, 1.0)
    return Image(path, iw * scale, ih * scale)


def parse_markdown(md_text, base_dir, doc_width):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleStyle",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1Style",
            parent=styles["Heading1"],
            fontSize=14,
            leading=18,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2Style",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyStyle",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionStyle",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            spaceAfter=10,
            textColor=colors.grey,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeStyle",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8.5,
            leading=10.5,
            spaceAfter=8,
        )
    )

    story = []
    lines = md_text.splitlines()
    paragraph_lines = []
    bullet_lines = []
    code_lines = []
    in_code = False

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(paragraph_lines)
            story.append(Paragraph(text, styles["BodyStyle"]))
            story.append(Spacer(1, 6))
            paragraph_lines = []

    def flush_bullets():
        nonlocal bullet_lines
        if bullet_lines:
            for bullet in bullet_lines:
                story.append(Paragraph(bullet, styles["BodyStyle"], bulletText="-"))
            story.append(Spacer(1, 4))
            bullet_lines = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            code_text = "\n".join(code_lines)
            story.append(Preformatted(code_text, styles["CodeStyle"]))
            code_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                flush_bullets()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if line.startswith("#"):
            flush_paragraph()
            flush_bullets()
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            if level == 1:
                story.append(Paragraph(text, styles["TitleStyle"]))
            elif level == 2:
                story.append(Paragraph(text, styles["Heading1Style"]))
            else:
                story.append(Paragraph(text, styles["Heading2Style"]))
            story.append(Spacer(1, 6))
            continue

        if line.startswith("![") and "](" in line and line.endswith(")"):
            flush_paragraph()
            flush_bullets()
            caption = line.split("![", 1)[1].split("]", 1)[0]
            path = line.split("(", 1)[1].rsplit(")", 1)[0]
            img_path = os.path.abspath(os.path.join(base_dir, path))
            if os.path.exists(img_path):
                story.append(scaled_image(img_path, doc_width))
                if caption:
                    story.append(Paragraph(caption, styles["CaptionStyle"]))
            continue

        if line.strip() == "":
            flush_paragraph()
            flush_bullets()
            continue

        if line.startswith("- "):
            bullet_lines.append(line[2:].strip())
            continue

        paragraph_lines.append(line.strip())

    flush_paragraph()
    flush_bullets()
    flush_code()

    return story


def build_pdf(md_text):
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )
    story = parse_markdown(md_text, BASE_DIR, doc.width)
    doc.build(story)


def main():
    if not os.path.exists(MODEL_METRICS_PATH):
        raise FileNotFoundError(f"Missing model metrics: {MODEL_METRICS_PATH}")
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    rows = read_model_metrics(MODEL_METRICS_PATH)
    counts, shifts, avg_shifts = read_results_counts(RESULTS_PATH)

    ensure_diagrams()
    md_text = build_markdown(rows, counts, avg_shifts)
    write_markdown(md_text)
    build_pdf(md_text)


if __name__ == "__main__":
    main()
