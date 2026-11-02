"""
For each judge-output file, report what share of the trained model's losses
come with a format violation by the trained model.

Usage: python format_loss_breakdown.py raw_outputs2_*_sum.jsonl
"""
import collections
import json
import sys

for path in sys.argv[1:]:
    c = collections.Counter()
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                c["malformed_lines"] += 1
                continue
            if d["verdict"] not in ("A", "B"):  # "Invalid" / "Oversized" judge outputs
                c["judge_invalid"] += 1
                continue
            c["valid"] += 1
            if d["trained_won"]:
                c["wins"] += 1
                continue
            c["losses"] += 1
            t, b = d["trained_format_violation"], d["base_format_violation"]
            if t is None or b is None:
                c["loss_unknown"] += 1
            elif t and b:
                c["loss_both"] += 1
            elif t:
                c["loss_trained_only"] += 1
            elif b:
                c["loss_base_only"] += 1
            else:
                c["loss_neither"] += 1

    L = c["losses"]
    fmt = c["loss_trained_only"] + c["loss_both"]
    print(f"{path}")
    print(f"  valid comparisons {c['valid']}, judge-invalid {c['judge_invalid']}, malformed lines {c['malformed_lines']}")
    print(f"  win rate {c['wins'] / c['valid']:.3f}, losses {L}")
    print(f"  losses with trained format violation: {fmt / L:.1%} "
          f"(trained only {c['loss_trained_only'] / L:.1%}, both {c['loss_both'] / L:.1%})")
    print(f"  losses with no violation: {c['loss_neither'] / L:.1%}, base-only violation: {c['loss_base_only'] / L:.1%}, "
          f"flags missing: {c['loss_unknown'] / L:.1%}")
    print(f"  win rate excluding format-violation losses: {c['wins'] / (c['valid'] - fmt):.3f}")