"""Audit readiness for the next independent-trajectory validation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kefir_models.prospective_validation import audit_plan, freeze_protocol, verify_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/prospective_validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/prospective/readiness.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--freeze", type=Path, help="Create a new protocol snapshot if metadata are complete."
    )
    mode.add_argument(
        "--verify", type=Path, help="Verify a previously frozen snapshot and model sources."
    )
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_protocol(args.verify), indent=2))
        return
    report = audit_plan(args.plan, inspect_prefixes=args.freeze is None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.freeze:
        freeze_protocol(args.plan, args.freeze)


if __name__ == "__main__":
    main()
