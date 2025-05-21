#!/usr/bin/env python3
"""
collect_slurm_stats_fixed.py — records live Slurm usage to a CSV with a *stable* GPU
schema so downstream analysis is painless.  The GPU columns are always written in a
fixed order, regardless of which cards are in use when the script runs.

Changes compared with the original version
------------------------------------------
* Added GPU_CATEGORIES / GPU_TYPES_ORDERED constants that enumerate every GPU type
  we care about.  The column layout never fluctuates.
* Removed the dynamic discovery of GPU types; the header is now deterministic.
* Normalised GPU type keys to lowercase to match our constant list.
* Defensive coding touches (stripping whitespace, defaulting to 0, etc.).

Drop‑in replacement — invoke from cron or a systemd timer exactly as before.
"""

import subprocess
import csv
from datetime import datetime
import os
from collections import defaultdict
import logging
import re

# --- GPU configuration (static schema) ---------------------------------------
GPU_CATEGORIES = {
    "gpu-low": [
        "titanx",
        "titanxp",
        "1080ti",
        "2080ti",
        "t4",
    ],
    "gpu-mid": [
        "titanrtx",
        "r6000",
        "l4",
        "3090",
        "a5000",
        "a5500",
        "v100",
    ],
    "gpu-high": [
        "a40",
        "a6000",
        "6000ada",
        "a100",
        "h100",
    ],
}

# Deterministic flat list for CSV columns
GPU_TYPES_ORDERED = [gpu for tier in ("gpu-low", "gpu-mid", "gpu-high") for gpu in GPU_CATEGORIES[tier]]

# --- Configuration -----------------------------------------------------------
OUTPUT_DIR = os.path.expanduser("~/slurm_monitor")  # Store data in user's home
DATA_FILE = os.path.join(OUTPUT_DIR, "slurm_resource_usage.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "slurm_collector.log")
SACCT_CMD = (
    "/usr/local/slurm/current/bin/sacct -a -X --format=User,AllocTRES,State "
    "--parsable2 --noheader --state=RUNNING"
)

# --- Logging Setup -----------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_mem_to_gb(mem_str: str) -> float:
    """Convert Slurm memory strings (e.g. "500G", "32000M") into GB as float."""
    if not mem_str:
        return 0.0
    match = re.fullmatch(r"([0-9.]+)([KMG]?)", mem_str.strip(), re.I)
    if not match:
        logging.error("Unrecognised mem format: %s", mem_str)
        return 0.0
    value, unit = match.groups()
    value = float(value)
    unit = unit.upper()
    if unit == "G":
        return value
    if unit == "M":
        return value / 1024.0
    if unit == "K":
        return value / (1024.0 * 1024.0)
    # Bytes (no unit)
    return value / (1024.0 ** 3)


def parse_alloc_tres(tres_string: str):
    print(tres_string)
    """Parse the AllocTRES field and return a structured dict."""
    resources = {
        "cpu": 0,
        "mem_gb": 0.0,
        "gpus": defaultdict(int),
        "total_gpus": 0,
    }

    if not tres_string or tres_string.lower() == "none":
        return resources

    for part in tres_string.split(","):
        if "=" not in part:
            continue
        key, value = (s.strip() for s in part.split("=", 1))
        if key == "cpu":
            resources["cpu"] += int(value)
        elif key == "mem":
            resources["mem_gb"] += parse_mem_to_gb(value)
        elif key.startswith("gres/gpu:"):
            gpu_type = key.split(":", 1)[1].lower()
            gpu_count = int(value)
            resources["gpus"][gpu_type] += gpu_count
            resources["total_gpus"] += gpu_count

    return resources


# -----------------------------------------------------------------------------
# Core collection logic
# -----------------------------------------------------------------------------

def get_slurm_usage():
    """Query sacct and aggregate usage per user."""
    usage = defaultdict(
        lambda: {
            "cpu": 0,
            "mem_gb": 0.0,
            "gpus": defaultdict(int),
            "total_gpus": 0,
            "job_count": 0,
        }
    )

    try:
        proc = subprocess.Popen(
            SACCT_CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logging.error("sacct failed: %s", stderr.strip())
            return {}
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                user, tres, state = line.split("|", 2)
            except ValueError:
                logging.warning("Malformed line from sacct: %s", line)
                continue
            if state.strip().upper() != "RUNNING":
                continue  # --state=RUNNING should guarantee this, but be safe

            tres_parsed = parse_alloc_tres(tres)
            rec = usage[user]
            rec["cpu"] += tres_parsed["cpu"]
            rec["mem_gb"] += tres_parsed["mem_gb"]
            rec["total_gpus"] += tres_parsed["total_gpus"]
            rec["job_count"] += 1
            for gtype, cnt in tres_parsed["gpus"].items():
                rec["gpus"][gtype] += cnt
    except FileNotFoundError:
        logging.error("sacct not found; ensure Slurm CLI tools are installed and in PATH")
    except Exception as exc:
        logging.exception("Unexpected error while collecting Slurm usage: %s", exc)

    return usage


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    logging.info("collect_slurm_stats_fixed starting an iteration")
    usage = get_slurm_usage()
    if not usage:
        logging.info("No running jobs; nothing to log this cycle")
        return

    timestamp = datetime.now().isoformat()
    fieldnames = (
        ["timestamp", "user", "job_count", "cpus", "mem_gb", "total_gpus"]
        + [f"gpu_{gpu}" for gpu in GPU_TYPES_ORDERED]
    )

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    try:
        with open(DATA_FILE, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            if csvfile.tell() == 0:
                writer.writeheader()

            for user, data in usage.items():
                row = {
                    "timestamp": timestamp,
                    "user": user,
                    "job_count": data["job_count"],
                    "cpus": data["cpu"],
                    "mem_gb": round(data["mem_gb"], 2),
                    "total_gpus": data["total_gpus"],
                }
                # Populate each GPU column, defaulting to 0
                for gpu_name in GPU_TYPES_ORDERED:
                    row[f"gpu_{gpu_name}"] = data["gpus"].get(gpu_name, 0)
                writer.writerow(row)
        logging.info("Wrote usage for %d user(s) to %s", len(usage), DATA_FILE)
    except IOError as exc:
        logging.error("Failed to write CSV: %s", exc)


if __name__ == "__main__":
    main()
