#!/usr/bin/env python3
"""Compile the reproducible many-to-one grid into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MEMBERS = (2, 4, 8, 16, 32, 64, 128)
ARCHITECTURES = ("2a", "fakv")
SEEDS = (20, 21, 22)


def mean_sd(values: list[float]) -> str:
    return f"{statistics.mean(values) * 100:.2f}% ± {statistics.stdev(values) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for members in MEMBERS:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                path = args.input / f"m{members}" / architecture / f"seed_{seed}" / "summary.json"
                summary = json.loads(path.read_text())
                metric = summary["test"] if architecture == "2a" else summary["hard_test"]
                row = {
                    "members": members,
                    "model": "2A" if architecture == "2a" else "FA-KV",
                    "seed": seed,
                    "parameters": summary["parameters"],
                    "test_accuracy": metric["accuracy"],
                    "test_loss": metric["loss"],
                }
                if architecture == "fakv":
                    learned = summary["learned_thresholds"]
                    row["hard_past_window"] = json.dumps(learned["hard_past"])
                    row["hard_future_window"] = json.dumps(learned["hard_future"])
                else:
                    row["hard_past_window"] = ""
                    row["hard_future_window"] = ""
                rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "| m | 2A test accuracy | FA-KV hard-window test accuracy |",
        "|---:|---:|---:|",
    ]
    for members in MEMBERS:
        values = {
            model: [row["test_accuracy"] for row in rows
                    if row["members"] == members and row["model"] == model]
            for model in ("2A", "FA-KV")
        }
        table.append(f"| {members} | {mean_sd(values['2A'])} | {mean_sd(values['FA-KV'])} |")

    generated = "\n".join(table)
    text = args.report.read_text()
    start, end = "<!-- generated-results:start -->", "<!-- generated-results:end -->"
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    args.report.write_text(f"{before}{start}\n{generated}\n{end}{after}")


if __name__ == "__main__":
    main()
