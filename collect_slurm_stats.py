#!/usr/bin/env python3

import subprocess
import re
import csv
from datetime import datetime
import os
from collections import defaultdict
import logging

# --- Configuration ---
OUTPUT_DIR = os.path.expanduser("~/slurm_monitor") # Store data in user's home
DATA_FILE = os.path.join(OUTPUT_DIR, "slurm_resource_usage.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "slurm_collector.log")
SACCT_CMD_FORMAT = "/usr/local/slurm/current/bin/sacct -a -X --format=User,AllocTRES,State --parsable2 --noheader --state=RUNNING"

# --- Logging Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_mem_to_gb(mem_str):
    """Converts memory string (e.g., 500G, 32000M, 102400K) to GB."""
    if not mem_str:
        return 0.0
    val_str = mem_str[:-1]
    unit = mem_str[-1:].upper()
    try:
        val = float(val_str)
        if unit == 'G':
            return val
        elif unit == 'M':
            return val / 1024.0
        elif unit == 'K':
            return val / (1024.0 * 1024.0)
        else: # Assuming bytes if no unit or unknown unit
            return float(mem_str) / (1024.0 * 1024.0 * 1024.0)
    except ValueError:
        logging.error(f"Could not parse memory value from: {mem_str}")
        return 0.0

def parse_alloc_tres(tres_string):
    """
    Parses the AllocTRES string.
    Example: billing=16,cpu=16,gres/gpu:a6000=4,gres/gpu=4,mem=500G,node=1
    Returns a dictionary: {'cpu': X, 'mem_gb': Y, 'gpus': {'type1': Z1, 'type2': Z2}, 'total_gpus': T}
    """
    resources = {'cpu': 0, 'mem_gb': 0.0, 'gpus': defaultdict(int), 'total_gpus': 0}
    if not tres_string or tres_string.lower() == 'none':
        return resources

    parts = tres_string.split(',')
    for part in parts:
        if not part:
            continue
        try:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()

            if key == 'cpu':
                resources['cpu'] += int(value)
            elif key == 'mem':
                resources['mem_gb'] += parse_mem_to_gb(value)
            elif key.startswith('gres/gpu:'):
                gpu_type = key.split(':')[1]
                gpu_count = int(value)
                resources['gpus'][gpu_type] += gpu_count
                resources['total_gpus'] += gpu_count
            elif key == 'gres/gpu': # This is a fallback if only total GPUs are listed
                                    # and not already counted by specific types
                if not resources['gpus']: # Only use if no specific types found
                     resources['total_gpus'] += int(value)
                     resources['gpus']['unknown'] += int(value)


        except ValueError as e:
            logging.warning(f"Could not parse TRES part '{part}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error parsing TRES part '{part}': {e}")
    return resources

def get_slurm_usage():
    """
    Runs sacct and aggregates resource usage by user.
    Returns a dictionary:
    {
        'user1': {'cpu': X, 'mem_gb': Y, 'gpus': {'typeA': N, 'typeB': M}, 'total_gpus': P, 'job_count': J},
        ...
    }
    """
    user_usage = defaultdict(lambda: {
        'cpu': 0,
        'mem_gb': 0.0,
        'gpus': defaultdict(int),
        'total_gpus': 0,
        'job_count': 0
    })

    try:
        process = subprocess.Popen(SACCT_CMD_FORMAT, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logging.error(f"sacct command failed with error: {stderr}")
            return {}
        if not stdout:
            logging.info("sacct command returned no output.")
            return {}

        lines = stdout.strip().split('\n')
        if not lines or (len(lines) == 1 and not lines[0].strip()): # Handle empty output after strip
            logging.info("No running jobs found.")
            return {}

        for line in lines:
            if not line.strip():
                continue
            fields = line.split('|')
            if len(fields) < 3: # Expecting User, AllocTRES, State
                logging.warning(f"Skipping malformed line: {line}")
                continue

            user, tres_string, state = fields[0], fields[1], fields[2]

            # We already filter by state=RUNNING in sacct, but double check
            if state.strip().upper() != "RUNNING":
                continue

            parsed_tres = parse_alloc_tres(tres_string)

            user_usage[user]['cpu'] += parsed_tres['cpu']
            user_usage[user]['mem_gb'] += parsed_tres['mem_gb']
            for gpu_type, count in parsed_tres['gpus'].items():
                user_usage[user]['gpus'][gpu_type] += count
            user_usage[user]['total_gpus'] += parsed_tres['total_gpus']
            user_usage[user]['job_count'] += 1

    except FileNotFoundError:
        logging.error("sacct command not found. Ensure Slurm tools are in PATH.")
        return {}
    except Exception as e:
        logging.error(f"An error occurred while running/parsing sacct: {e}")
        return {}

    return user_usage

def main():
    logging.info("Starting data collection.")
    current_usage = get_slurm_usage()
    timestamp = datetime.now().isoformat()

    if not current_usage:
        logging.info("No usage data collected.")
        return

    all_gpu_types = set()
    for data in current_usage.values():
        all_gpu_types.update(data['gpus'].keys())
    sorted_gpu_types = sorted(list(all_gpu_types))

    fieldnames = ['timestamp', 'user', 'job_count', 'cpus', 'mem_gb', 'total_gpus'] + \
                 [f'gpu_{gtype}' for gtype in sorted_gpu_types]

    file_exists = os.path.isfile(DATA_FILE)
    try:
        with open(DATA_FILE, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(DATA_FILE) == 0:
                writer.writeheader()

            for user, data in current_usage.items():
                row = {
                    'timestamp': timestamp,
                    'user': user,
                    'job_count': data['job_count'],
                    'cpus': data['cpu'],
                    'mem_gb': round(data['mem_gb'], 2),
                    'total_gpus': data['total_gpus']
                }
                for gpu_type in sorted_gpu_types:
                    row[f'gpu_{gpu_type}'] = data['gpus'].get(gpu_type, 0)
                writer.writerow(row)
        logging.info(f"Successfully wrote {len(current_usage)} user(s) data to {DATA_FILE}")
    except IOError as e:
        logging.error(f"Could not write to {DATA_FILE}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during file writing: {e}")

if __name__ == "__main__":
    main()