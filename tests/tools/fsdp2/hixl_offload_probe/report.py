# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Append-only records and portable feedback bundle; no hardware dependency."""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tarfile
import time


def append_json(path, record):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": time.time(), **record}, ensure_ascii=False) + "\n")
        stream.flush()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def redirect_logs(path):
    # Capture native CANN/HIXL output as well as Python tracebacks.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    faulthandler.enable()


def read_rows(path):
    rows = []
    if Path(path).exists():
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"status": "FAIL", "case": "artifact", "phase": "parse",
                             "error": f"truncated/invalid JSONL line {line_no}"})
    return rows


def build_report(directory):
    directory = Path(directory)
    rows = read_rows(directory / "results.jsonl")
    # The parent copies worker rows only after a case exits. Recover an interrupted
    # case from its own journal when the runner was killed before that copy.
    completed = {(row.get("case"), row.get("mode")) for row in rows
                 if row.get("phase") == "case_result"}
    for case_dir in sorted((directory / "cases").glob("*")):
        case, _, mode = case_dir.name.rpartition("_")
        if (case, mode) not in completed:
            rows.extend(read_rows(case_dir / "results.jsonl"))
            rows.append({"case": case, "mode": mode, "phase": "case_result", "status": "FAIL",
                         "reason": "runner did not record case completion; recovered partial worker journal"})
    runs = [row for row in rows if row.get("phase") in ("case_result", "preflight")]
    counts = {key: sum(row.get("status") == key for row in runs)
              for key in ("PASS", "FAIL", "UNSUPPORTED", "SKIP")}
    lines = ["# HIXL activation offload probe", "",
             "## Run outcomes", "", "```json", json.dumps(counts, indent=2), "```", "",
             "| Case | Mode | Status | Reason |", "|---|---|---|---|"]
    for row in runs:
        reason = str(row.get("error", row.get("reason", ""))).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row.get('case', 'preflight')} | {row.get('mode', '-')} | {row['status']} | {reason} |")
    lines += ["", "## Transfer timings (verification excluded)", "",
              "Sync submit time includes completion. Async completion includes Host polling latency.", "",
              "| Case | Mode | Bytes per buffer | Direction | Samples | Median ms | p95 ms | MiB/s |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    groups = {}
    measured = [row for row in rows if row.get("phase") == "iteration" and not row.get("warmup")]
    for row in measured:
        for direction in ("write", "read"):
            key = (row["case"], row["mode"], row["size_bytes"], direction)
            groups.setdefault(key, []).append(row[direction])
    for (case, mode, size, direction), items in sorted(groups.items()):
        times = sorted(item["complete_ms"] for item in items)
        median = statistics.median(times)
        p95 = times[max(0, (95 * len(times) + 99) // 100 - 1)]
        bandwidth = sum(item["bytes"] for item in items) / (sum(times) / 1000) / (1024 ** 2) if sum(times) else 0
        lines.append(f"| {case} | {mode} | {size} | {direction} | {len(items)} | {median:.3f} | {p95:.3f} | {bandwidth:.2f} |")
    lines += ["", "## Dynamic-memory observations", "",
              "| Case | Mode | Bytes per buffer | Median allocated drop after source release (bytes) | Last allocated minus first baseline (bytes) |",
              "|---|---|---:|---:|---:|"]
    memory_groups = {}
    for row in measured:
        if row.get("registration_timing") == "after_connect":
            key = (row["case"], row["mode"], row["size_bytes"])
            memory_groups.setdefault(key, []).append(row)
    for (case, mode, size), items in sorted(memory_groups.items()):
        drops = [item["memory_source_live"]["allocated_bytes"] - item["memory_source_released"]["allocated_bytes"]
                 for item in items]
        growth = items[-1]["memory_after"]["allocated_bytes"] - items[0]["memory_before"]["allocated_bytes"]
        lines.append(f"| {case} | {mode} | {size} | {statistics.median(drops):.0f} | {growth} |")
    reused = any(row.get("address_reused_after_release") or row.get("source_reused_across_iterations")
                 for row in measured if row.get("registration_timing") == "after_connect")
    lines += ["", "## Coverage and interpretation", "",
              f"- Allocator address reuse observed: **{'yes' if reused else 'no — coverage incomplete'}**.",
              "- Every passing iteration verifies WRITE on the pool and READ on the client using SHA-256 of a position-dependent pattern.",
              "- `application_staging=false`: this probe adds no activation staging copy. CPU initialization/checking is outside transfer timing.",
              "- Actual protocol, NotifyWait/ValueWait and Host exception callback: **unknown unless separately evidenced by runtime logs**.",
              "- Library-internal zero staging: **unverified**. No inference from API success or version alone.",
              "- PyTorch reserved memory may remain cached. Inspect allocated memory, registration counts and ACL allocations separately.",
              "- End-to-end iteration timings include CPU pattern generation, control RPCs and verification; they are not a training-throughput estimate.",
              "- A failure stops that isolated case. Other cases run in fresh processes; no protocol or registration fallback is applied.",
              "", "## Diagnostics", ""]
    for row in rows:
        if row.get("phase") in ("failure", "pool_failure", "parse"):
            lines.append(f"- {row.get('case', 'pool')} / {row.get('mode', '-')}: {row.get('error')}")
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bundle(directory):
    directory = Path(directory)
    # Only generated artifacts; never recursively collect external CANN logs/env.
    paths = sorted(path for path in directory.rglob("*") if path.is_file()
                   and not path.is_symlink() and path.suffix in (".json", ".jsonl", ".log", ".md"))
    checksums = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in paths if path.name != "manifest.json"}
    write_json(directory / "manifest.json", checksums)
    paths = [p for p in paths if p.name != "manifest.json"] + [directory / "manifest.json"]
    with tarfile.open(directory / "artifacts.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(directory)), recursive=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate probe summary and feedback archive")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_report(args.output)
    bundle(args.output)
