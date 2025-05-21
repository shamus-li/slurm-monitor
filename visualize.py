#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For formatting dates on the x-axis
import seaborn as sns
import os
from datetime import datetime
from collections import defaultdict

# --- Configuration ---
OUTPUT_DIR = os.path.expanduser("~/slurm_monitor")
DATA_FILE = os.path.join(OUTPUT_DIR, "slurm_resource_usage.csv")
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "plots")

# User-defined GPU ranking (higher value means "better")
GPU_RANKING = {
    # High-end / Latest
    'h200': 22,
    'h100': 21,
    'a100': 20,
    'a6000': 19,
    '6000ada': 18, # RTX 6000 Ada Generation
    'l40s': 17,
    'l4': 16,
    # Mid-to-High end / Previous Gen
    'v100': 15,
    'a40': 14,
    'a5000': 13,
    'rtx8000': 12, # Quadro RTX 8000
    'rtx6000': 11, # Quadro RTX 6000
    # Consumer High-end
    '4090': 10,
    '3090': 9,
    'titanrtx': 8,
    'titanxp': 7,
    '2080ti': 6,
    # Other
    'a5500': 5,
    '1080ti': 4,
    'titanx': 3, # Older Titan X
    'p100': 2,
    'k80': 1,
    'unknown': 0 # For any GPUs not in this list
}


# --- Ensure plot directory exists ---
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

def get_gpu_ranking_key(gpu_col_name):
    """Helper function to get sorting key for GPU columns based on GPU_RANKING."""
    gpu_type = gpu_col_name.replace('gpu_', '') # e.g., 'gpu_a100' -> 'a100'
    return GPU_RANKING.get(gpu_type.lower(), GPU_RANKING['unknown'])


def generate_visualizations():
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        print(f"Data file {DATA_FILE} not found or is empty. Run the collector first.")
        return

    try:
        df = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        print(f"Data file {DATA_FILE} is empty or corrupted.")
        return
    except Exception as e:
        print(f"Error reading data file {DATA_FILE}: {e}")
        return

    if df.empty:
        print("No data to visualize.")
        return

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error converting 'timestamp' column to datetime: {e}")
        print("Please ensure timestamps in the CSV are in a recognizable format.")
        return

    # --- Plot 0a: Total GPU Usage Over Time (Cluster-wide) ---
    if 'total_gpus' in df.columns:
        gpu_over_time_cluster = df.groupby('timestamp')['total_gpus'].sum().reset_index()
        if not gpu_over_time_cluster.empty:
            plt.figure(figsize=(18, 7))
            plt.plot(gpu_over_time_cluster['timestamp'], gpu_over_time_cluster['total_gpus'], marker='o', linestyle='-', label="Cluster Total")
            plt.title('Total GPU Usage Over Time (Cluster-wide)')
            plt.xlabel('Timestamp')
            plt.ylabel('Total GPUs Used')
            plt.grid(True)
            plt.xticks(rotation=45)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
            plt.legend()
            plt.tight_layout()
            plot_filename_gpu_time_cluster = os.path.join(PLOT_OUTPUT_DIR, f"cluster_gpu_usage_over_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_filename_gpu_time_cluster)
            plt.close()
            print(f"Saved cluster-wide GPU usage over time plot to {plot_filename_gpu_time_cluster}")
        else:
            print("No 'total_gpus' data to plot for cluster-wide usage over time.")
    else:
        print("Column 'total_gpus' not found. Cannot generate cluster-wide GPU usage over time plot.")

    # --- Plot 0b: GPU Usage Over Time (Top 10 Users) ---
    if 'total_gpus' in df.columns and 'user' in df.columns:
        # Calculate total GPU-hours or a similar metric for ranking if timestamps are regular
        # For simplicity, we'll sum total_gpus recorded for each user as a proxy for overall usage.
        total_gpu_usage_per_user = df.groupby('user')['total_gpus'].sum()
        
        top_10_gpu_users = total_gpu_usage_per_user.nlargest(10).index.tolist()
        
        if not top_10_gpu_users:
            print("No users with GPU usage found to determine top 10 for time-series plot.")
        else:
            print(f"Top 10 GPU users (overall for time-series plot): {top_10_gpu_users}")

            user_gpu_over_time_pivot_all = df.pivot_table(index='timestamp', columns='user', values='total_gpus', aggfunc='sum').fillna(0)
            
            # Filter for top 10 users that are actually in the columns
            cols_to_plot = [user for user in top_10_gpu_users if user in user_gpu_over_time_pivot_all.columns]

            if not cols_to_plot:
                print("None of the identified top 10 users are present in the pivoted data columns.")
            else:
                user_gpu_over_time_pivot_top10 = user_gpu_over_time_pivot_all[cols_to_plot]

                if not user_gpu_over_time_pivot_top10.empty:
                    plt.figure(figsize=(18, 10))
                    for user_col in user_gpu_over_time_pivot_top10.columns:
                        plt.plot(user_gpu_over_time_pivot_top10.index, user_gpu_over_time_pivot_top10[user_col], marker='.', linestyle='-', label=user_col)
                    
                    plt.title('GPU Usage Over Time (Top 10 Users Overall)')
                    plt.xlabel('Timestamp')
                    plt.ylabel('Total GPUs Used')
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    ax = plt.gca()
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
                    plt.legend(title='Top 10 Users', bbox_to_anchor=(1, 1), loc='upper right')
                    plt.tight_layout()
                    plot_filename_gpu_time_user = os.path.join(PLOT_OUTPUT_DIR, f"top10_user_gpu_usage_over_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.savefig(plot_filename_gpu_time_user)
                    plt.close()
                    print(f"Saved top 10 per-user GPU usage over time plot to {plot_filename_gpu_time_user}")
                else:
                    print("Pivoted data for top 10 users is empty.")
    else:
        print("Columns 'total_gpus' or 'user' not found. Cannot generate per-user GPU usage over time plot.")

    # --- Snapshot Plots (Latest Timestamp) ---
    latest_timestamp = df['timestamp'].max()
    latest_df = df[df['timestamp'] == latest_timestamp].copy()

    if latest_df.empty:
        print("No data for the latest timestamp to generate snapshot plots.")
        return

    all_snapshot_gpu_cols = [col for col in latest_df.columns if col.startswith('gpu_') and col != 'total_gpus']
    if 'total_gpus' not in latest_df.columns and all_snapshot_gpu_cols:
         latest_df['total_gpus'] = latest_df[all_snapshot_gpu_cols].sum(axis=1)
    elif 'total_gpus' not in latest_df.columns:
        latest_df['total_gpus'] = 0

    latest_df_sorted = latest_df.sort_values(by=['total_gpus', 'user'], ascending=[False, True])

    # --- Plot 1: Total GPU Usage per User (Snapshot) ---
    if 'total_gpus' in latest_df_sorted.columns:
        plt.figure(figsize=(18, 8))
        sns.barplot(x='user', y='total_gpus', data=latest_df_sorted, palette="viridis")
        plt.title(f'Total GPU Usage per User (Snapshot: {latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")})')
        plt.xlabel('User')
        plt.ylabel('Total GPUs Used')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename_gpu = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_gpu_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename_gpu)
        plt.close()
        print(f"Saved snapshot GPU usage plot to {plot_filename_gpu}")
    else:
        print("Column 'total_gpus' not found in latest snapshot. Cannot generate snapshot GPU usage plot.")

    # --- Plot 2: Stacked GPU Usage by Type per User (Snapshot, if GPU types exist) ---
    snapshot_gpu_type_cols = [col for col in latest_df_sorted.columns if col.startswith('gpu_') and col != 'total_gpus']

    if snapshot_gpu_type_cols:
        sorted_gpu_type_cols = sorted(snapshot_gpu_type_cols, key=get_gpu_ranking_key)

        gpu_users_df = latest_df_sorted[latest_df_sorted['total_gpus'] > 0][['user'] + sorted_gpu_type_cols].copy()
        gpu_users_df.set_index('user', inplace=True)

        if not gpu_users_df.empty:
            gpu_users_df = gpu_users_df[sorted_gpu_type_cols] # Ensure correct column order for stacking

            gpu_users_df.plot(kind='bar', stacked=True, figsize=(18, 10), colormap="rainbow")
            plt.title(f'GPU Usage by Type per User (Snapshot: {latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")})')
            plt.xlabel('User (Sorted by Total GPU Usage)')
            plt.ylabel('Number of GPUs')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='GPU Type', loc='upper right')
            plt.tight_layout()
            plot_filename_stacked_gpu = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_stacked_gpu_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_filename_stacked_gpu)
            plt.close()
            print(f"Saved snapshot stacked GPU usage plot to {plot_filename_stacked_gpu}")
        else:
            print("No users currently using GPUs in the latest snapshot to generate stacked plot.")
    else:
        print("No specific GPU type columns (e.g., 'gpu_a100') found. Skipping snapshot stacked GPU plot.")

    # --- Plot 3: CPU Usage per User (Snapshot, Sorted by GPU usage) ---
    if 'cpus' in latest_df_sorted.columns:
        plt.figure(figsize=(18, 8)) 
        sns.barplot(x='user', y='cpus', data=latest_df_sorted, palette="coolwarm")
        plt.title(f'CPU Usage per User (Snapshot: {latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")}, Sorted by GPU Usage)')
        plt.xlabel('User')
        plt.ylabel('CPUs Used')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename_cpu = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_cpu_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename_cpu)
        plt.close()
        print(f"Saved snapshot CPU usage plot to {plot_filename_cpu}")
    else:
        print("Column 'cpus' not found in latest snapshot. Cannot generate snapshot CPU usage plot.")

    # --- Plot 4: Memory Usage per User (Snapshot, Sorted by GPU usage) ---
    if 'mem_gb' in latest_df_sorted.columns:
        plt.figure(figsize=(18, 8))
        sns.barplot(x='user', y='mem_gb', data=latest_df_sorted, palette="autumn")
        plt.title(f'Memory Usage (GB) per User (Snapshot: {latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")}, Sorted by GPU Usage)')
        plt.xlabel('User')
        plt.ylabel('Memory Used (GB)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename_mem = os.path.join(PLOT_OUTPUT_DIR, f"snapshot_memory_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename_mem)
        plt.close()
        print(f"Saved snapshot Memory usage plot to {plot_filename_mem}")
    else:
        print("Column 'mem_gb' not found in latest snapshot. Cannot generate snapshot memory usage plot.")

if __name__ == "__main__":
    generate_visualizations()
