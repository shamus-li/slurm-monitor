#!/usr/bin/env python3
"""
visualize_gpu_usage.py — companion to *collect_slurm_stats_fixed.py*
-------------------------------------------------------------------
This version recognises the **gpu-low / gpu-mid / gpu-high** tiers that the
collector now uses for its stable CSV schema.  Functionality is otherwise
unchanged: it reads `slurm_resource_usage.csv` and emits the same suite of
PNG plots.  The only differences are

* a deterministic list of GPU types (`GPU_TYPES_ORDERED`) derived from the
  category spec, giving us column stability in every plot;
* an auto-generated `GPU_RANKING` so new cards can be added by simply
  updating the category lists; unrecognised cards fall into an "unknown"
  bucket but still plot;
* `get_gpu_ranking_key()` now uses the dynamic ranking list.

Run exactly as before:
    $ ./visualize_gpu_usage.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # For formatting dates on the x-axis
import seaborn as sns
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.expanduser("~/slurm_monitor")
DATA_FILE = os.path.join(OUTPUT_DIR, "slurm_resource_usage.csv")
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "plots")

# GPU categories (mirrors collect_slurm_stats_fixed)
GPU_CATEGORIES = {
    "gpu-low": [
        "t4",
        "2080ti",
        "1080ti",
        "titanxp",
        "titanx",
    ],
    "gpu-mid": [
        "v100",
        "a5500",
        "a5000",
        "3090",
        "l4",
        "r6000",
        "titanrtx",
    ],
    "gpu-high": [
        "h100",
        "a100",
        "6000ada",
        "a6000",
        "a40",
    ],
}

# Flattened, ordered list (high → low) + "unknown" catch-all
GPU_TYPES_ORDERED = [
    *GPU_CATEGORIES["gpu-high"],
    *GPU_CATEGORIES["gpu-mid"],
    *GPU_CATEGORIES["gpu-low"],
]

# Dynamically build a ranking: bigger number = higher tier
GPU_RANKING = {gpu: len(GPU_TYPES_ORDERED) - idx for idx, gpu in enumerate(GPU_TYPES_ORDERED)}
GPU_RANKING["unknown"] = 0  # explicitly lowest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gpu_ranking_key(gpu_col_name: str) -> int:
    """Return sort key based on GPU_RANKING dict (defaults to 0)."""
    gpu_type = gpu_col_name.replace("gpu_", "").lower()
    return GPU_RANKING.get(gpu_type, 0)


# ---------------------------------------------------------------------------
# Plotting routine (unchanged except for ranking logic)
# ---------------------------------------------------------------------------

def generate_visualizations():
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        print(f"Data file {DATA_FILE} not found or is empty. Run the collector first.")
        return

    try:
        df = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        print(f"Data file {DATA_FILE} is empty or corrupted.")
        return
    except Exception as exc:
        print(f"Error reading data file {DATA_FILE}: {exc}")
        return

    if df.empty:
        print("No data to visualize.")
        return

    # Ensure plot directory exists
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # Convert timestamp column
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as exc:
        print(f"Error converting 'timestamp' column to datetime: {exc}")
        return

    # -----------------------------
    # Plot 0a: Cluster total GPUs
    # -----------------------------
    if "total_gpus" in df.columns:
        cluster_gpu = df.groupby("timestamp")["total_gpus"].sum().reset_index()
        if not cluster_gpu.empty:
            plt.figure(figsize=(18, 7))
            plt.plot(cluster_gpu["timestamp"], cluster_gpu["total_gpus"], marker="o", linestyle="-", label="Cluster Total")
            plt.title("Total GPU Usage Over Time (Cluster-wide)")
            plt.xlabel("Timestamp")
            plt.ylabel("Total GPUs Used")
            plt.grid(True)
            plt.xticks(rotation=45)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
            plt.legend()
            plt.tight_layout()
            out = os.path.join(PLOT_OUTPUT_DIR, f"cluster_gpu_usage_over_time_{datetime.now():%Y%m%d_%H%M%S}.png")
            plt.savefig(out)
            plt.close()
            print(f"Saved cluster-wide GPU usage plot → {out}")
    else:
        print("Column 'total_gpus' not found — skipping cluster-wide GPU plot.")

    # -----------------------------
    # Plot 0b: Per-user time series (top 10 by GPU use)
    # -----------------------------
    if {"total_gpus", "user"}.issubset(df.columns):
        top_users = df.groupby("user")["total_gpus"].sum().nlargest(10).index
        pivot = df.pivot_table(index="timestamp", columns="user", values="total_gpus", aggfunc="sum").fillna(0)
        pivot = pivot[top_users.intersection(pivot.columns)]
        if not pivot.empty:
            plt.figure(figsize=(18, 10))
            for user in pivot.columns:
                plt.plot(pivot.index, pivot[user], marker=".", linestyle="-", label=user, linewidth=4, markersize=12)
            plt.title("GPU Usage Over Time (Top 10 Users)")
            plt.xlabel("Timestamp")
            plt.ylabel("Total GPUs Used")
            plt.grid(True)
            plt.xticks(rotation=45)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
            plt.legend(title="User", bbox_to_anchor=(1, 1), loc="upper right")
            plt.tight_layout()
            out = os.path.join(PLOT_OUTPUT_DIR, f"top10_user_gpu_usage_over_time_{datetime.now():%Y%m%d_%H%M%S}.png")
            plt.savefig(out)
            plt.close()
            print(f"Saved top-user time-series GPU plot → {out}")
    else:
        print("Required columns for per-user time-series plot not present.")

    # --------------------------------------------------------------------
    # Snapshot plots (latest timestamp)
    # --------------------------------------------------------------------
    latest_ts = df["timestamp"].max()
    snap = df[df["timestamp"] == latest_ts].copy()
    if snap.empty:
        print("No data for latest snapshot.")
        return

    gpu_cols = [c for c in snap.columns if c.startswith("gpu_") and c != "total_gpus"]
    if "total_gpus" not in snap.columns:
        snap["total_gpus"] = snap[gpu_cols].sum(axis=1)

    snap_sorted = snap.sort_values(["total_gpus", "user"], ascending=[False, True])

    # Plot 1: total GPUs per user (bar)
    if "total_gpus" in snap_sorted.columns:
        plt.figure(figsize=(18, 8))
        sns.barplot(x="user", y="total_gpus", data=snap_sorted, palette="viridis")
        plt.title(f"Total GPU Usage per User (Snapshot: {latest_ts:%Y-%m-%d %H:%M:%S})")
        plt.xlabel("User")
        plt.ylabel("Total GPUs Used")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_gpu_usage_{datetime.now():%Y%m%d_%H%M%S}.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved snapshot total GPU plot → {out}")

    # Plot 2: stacked GPUs by type per user
    if gpu_cols:
        ordered_cols = sorted(gpu_cols, key=get_gpu_ranking_key)
        stacked_df = snap_sorted[snap_sorted["total_gpus"] > 0][["user"] + ordered_cols].set_index("user")
        if not stacked_df.empty:
            stacked_df = stacked_df[ordered_cols]
            stacked_df.plot(kind="bar", stacked=True, figsize=(18, 10), colormap="winter")
            plt.title(f"GPU Usage by Type per User (Snapshot: {latest_ts:%Y-%m-%d %H:%M:%S})")
            plt.xlabel("User")
            plt.ylabel("GPUs Used")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="GPU Type", loc="upper right")
            plt.tight_layout()
            out = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_stacked_gpu_usage_{datetime.now():%Y%m%d_%H%M%S}.png")
            plt.savefig(out)
            plt.close()
            print(f"Saved snapshot stacked GPU plot → {out}")

    # Plot 3: CPU per user
    if "cpus" in snap_sorted.columns:
        plt.figure(figsize=(18, 8))
        sns.barplot(x="user", y="cpus", data=snap_sorted, palette="coolwarm")
        plt.title(f"CPU Usage per User (Snapshot: {latest_ts:%Y-%m-%d %H:%M:%S})")
        plt.xlabel("User")
        plt.ylabel("CPUs Used")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_cpu_usage_{datetime.now():%Y%m%d_%H%M%S}.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved snapshot CPU plot → {out}")

    # Plot 4: Memory per user
    if "mem_gb" in snap_sorted.columns:
        plt.figure(figsize=(18, 8))
        sns.barplot(x="user", y="mem_gb", data=snap_sorted, palette="autumn")
        plt.title(f"Memory Usage (GB) per User (Snapshot: {latest_ts:%Y-%m-%d %H:%M:%S})")
        plt.xlabel("User")
        plt.ylabel("Memory (GB)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_memory_usage_{datetime.now():%Y%m%d_%H%M%S}.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved snapshot memory plot → {out}")


if __name__ == "__main__":
    generate_visualizations()
