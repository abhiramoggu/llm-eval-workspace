# visualize_topics.py
"""
Plot topic flow: user segments, CRS responses, recovery points.
Saves figures under config.FIG_DIR.
"""

import os
import matplotlib.pyplot as plt

from config import FIG_DIR
from evaluate import extract_topics, topics_to_text, system_alignment_with_topics


def plot_topic_flow(conversation, eval_detail, title="Topic Flow", save_as=None):
    """
    Visualize:
      - USER segments with their topic labels
      - CRS responses over time
      - Mark recovery points per shift (if any)
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    user_segments = eval_detail.get("user_segments", [])
    per_shift = eval_detail.get("per_shift", [])

    plt.figure(figsize=(12, 5))
    # Plot USER turns as top markers
    uy = [1 if t["speaker"] == "USER" else None for t in conversation]
    sy = [0 if t["speaker"] == "SYSTEM" else None for t in conversation]

    plt.scatter([i for i, v in enumerate(uy) if v is not None], [1]*sum(v is not None for v in uy), label="USER", s=30)
    plt.scatter([i for i, v in enumerate(sy) if v is not None], [0]*sum(v is not None for v in sy), label="SYSTEM", s=30)

    # Annotate SYSTEM turns with their extracted topics
    for i, turn in enumerate(conversation):
        if turn["speaker"] == "SYSTEM":
            sys_topics = extract_topics(turn["text"])
            if sys_topics:
                plt.text(i, -0.08, topics_to_text(sys_topics), ha="center", va="top", fontsize=8, color="darkblue", rotation=0)

    # Annotate user segments
    for seg in user_segments:
        mid = (seg["start_idx"] + seg["end_idx"]) / 2
        plt.axvspan(seg["start_idx"] - 0.4, seg["end_idx"] + 0.4, color="lightgray", alpha=0.3)
        plt.text(mid, 1.08, topics_to_text(seg["topics"]), ha="center", va="bottom", fontsize=9, rotation=0)

    # Mark recovery outcomes
    for d in per_shift:
        # recovery begins after new segment starts; we just annotate result
        to_topics = d["to_topics"]
        if d["recovered"]:
            label = f"REC✓ ({topics_to_text(to_topics)})"
            # approximate recovery x-position: start of new seg + delay * 2 (approx every 2 turns includes a SYSTEM)
            xpos = d["recovery_delay_sys_turns"]
            # Try to place near the middle of new segment
            plt.text(seg["end_idx"] + (xpos or 1), 0.1, label, color="green", fontsize=9)
        else:
            label = f"REC✗ ({topics_to_text(to_topics)})"
            plt.text(seg["end_idx"] + 1, 0.1, label, color="red", fontsize=9)

    plt.yticks([0, 1], ["SYSTEM", "USER"])
    plt.ylim(-0.3, 1.3)
    plt.xlabel("Turn")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_as is None:
        save_as = os.path.join(FIG_DIR, "topic_flow.png")
    plt.savefig(save_as, dpi=160)
    plt.close()
