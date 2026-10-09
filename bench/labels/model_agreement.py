"""How far human labels agree with model suggestions, split by whether the human saw them.

- blind: points in the seeded blind subset, labelled without any suggestion
  shown. Only this split gives an agreement rate, and only from MIN_RATE_N
  points up, since it is the only one the model could not have anchored.
- anchored: points labelled with the model's suggestion on screen; counted
  as accepted or overridden, never as agreement.
- unseen: other points labelled without a suggestion (for instance before
  the model ran); counted, not rated, since they weren't chosen at random.
"""
from __future__ import annotations

MIN_RATE_N = 10


def model_agreement(entries: dict, usable: dict[str, str], model_data: dict) -> dict:
    """`entries`: the label file's points; `usable`: their non-stale labels; `model_data`: the model file."""
    blind_ids = set(model_data.get("blind_ids", []))
    suggested = {pid: e.get("suggested") for pid, e in model_data["points"].items() if e.get("suggested")}
    blind = [pid for pid in usable if pid in blind_ids and pid in suggested]
    agreed = sum(1 for pid in blind if usable[pid] == suggested[pid])
    anchored = [entries[pid].get("suggestion") for pid in usable if entries[pid].get("suggestion")]
    unseen = [pid for pid in usable if pid not in blind_ids and not entries[pid].get("suggestion")]
    return {
        "model": model_data.get("model"),
        "blind": {"agree": agreed, "points": len(blind),
                  "rate": round(agreed / len(blind), 4) if len(blind) >= MIN_RATE_N else None},
        "anchored": {"accepted": anchored.count("accepted"), "overridden": anchored.count("overridden")},
        "unseen": len(unseen),
    }
