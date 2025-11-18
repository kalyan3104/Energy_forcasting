#!/usr/bin/env python3
"""Simple live data feeder for the demo project.

This script appends synthetic power values to data/processed/sample.csv and
rewrites sample.parquet so the UI/backend see updated recent values. Run it in
the background during demos to simulate a live sensor.

Usage:
  python scripts/live_feeder.py --interval 5

"""
import argparse
import time
from pathlib import Path
import math
import random
import pandas as pd
from datetime import datetime, timedelta


def load_existing(path_csv, path_parquet):
    if path_csv.exists():
        try:
            return pd.read_csv(path_csv, parse_dates=["timestamp"])
        except Exception:
            pass
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet)
        except Exception:
            pass
    # fallback: generate a small baseline series
    now = datetime.utcnow()
    times = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
    powers = [0.5 + 0.1 * math.sin(i / 6.0) for i in range(len(times))]
    return pd.DataFrame({"timestamp": times, "power": powers})


def main(interval: int):
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "sample.csv"
    parquet_path = data_dir / "sample.parquet"

    df = load_existing(csv_path, parquet_path)
    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Starting live feeder (interval={interval}s). Writing to {csv_path}")
    try:
        while True:
            last_ts = df["timestamp"].iloc[-1]
            last_power = float(df["power"].iloc[-1])
            # next timestamp: add 1 minute
            next_ts = last_ts + timedelta(minutes=1)
            # synthetic next power: trend + daily-ish sin + noise
            t = len(df)
            trend = 0.0005 * t
            wave = 0.2 * math.sin(t / 12.0)
            noise = random.uniform(-0.02, 0.02)
            next_power = max(0.0, last_power + trend + wave + noise)
            new_row = {"timestamp": next_ts, "power": next_power}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # keep last 1440 minutes to avoid unbounded growth
            if len(df) > 1440:
                df = df.iloc[-1440:].reset_index(drop=True)

            # write CSV and parquet
            try:
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print("Failed to write CSV:", e)
            try:
                df.to_parquet(parquet_path, index=False)
            except Exception as e:
                print("Failed to write parquet:", e)

            print(f"Appended {next_ts.isoformat()} -> {next_power:.4f}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Feeder stopped by user")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=5, help="Seconds between appends")
    args = p.parse_args()
    main(args.interval)
