"""How far human labels agree with model suggestions, split by whether the human saw them.

- blind: labels made with no suggestion on screen, on a random set of points.
  Two sets qualify: the seeded blind subset, and a whole sample answered
  before any suggestion existed (`pre_suggestion`; the sample itself is a
  seeded random draw). Only this split gives an agreement rate, and only
  from MIN_RATE_N points up, since the model could not have anchored it.
- anchored: points labelled with the model's suggestion on screen; counted
  as accepted or overridden, never as agreement.
- unseen: other labels made without a suggestion, such as part of a sample
  labelled before the model ran. `label next` goes through a sample in
  point-ID order (by pull request), so a partly done sample is not a random
  subset: these are counted, not rated.
"""
from __future__ import annotations

MIN_RATE_N = 10


def answered_before_suggestions(entries: dict, sample_ids: list[str]) -> bool:
    """Every sampled point was labelled or skipped, and none was shown a suggestion."""
    return all(pid in entries for pid in sample_ids) and not any(entries[pid].get("suggestion") for pid in sample_ids)


def model_agreement(entries: dict, usable: dict[str, str], model_data: dict, sample_ids: list[str]) -> dict:
    """`entries`: the label file's points; `usable`: their non-stale labels in the sample; `model_data`: the
    model file; `sample_ids`: the label sample."""
    blind_ids = set(model_data.get("blind_ids", []))
    suggested = {pid: e.get("suggested") for pid, e in model_data["points"].items() if e.get("suggested")}
    pre = set(usable) - blind_ids if answered_before_suggestions(entries, sample_ids) else set()
    blind = [pid for pid in usable if (pid in blind_ids or pid in pre) and pid in suggested]
    agreed = sum(1 for pid in blind if usable[pid] == suggested[pid])
    anchored = [entries[pid].get("suggestion") for pid in usable if entries[pid].get("suggestion")]
    unseen = [pid for pid in usable if pid not in blind_ids and pid not in pre and not entries[pid].get("suggestion")]
    return {
        "model": model_data.get("model"),
        "blind": {"agree": agreed, "points": len(blind), "pre_suggestion": len([p for p in blind if p in pre]),
                  "rate": round(agreed / len(blind), 4) if len(blind) >= MIN_RATE_N else None},
        "anchored": {"accepted": anchored.count("accepted"), "overridden": anchored.count("overridden")},
        "unseen": len(unseen),
    }
