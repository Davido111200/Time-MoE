import pandas as pd
import json

in_path  = "/scratch/s223540177/Time-Series-Library/data/all_datasets/ETT-small/ETTh2.csv"
out_path = "/scratch/s223540177/Time-Series-Library/data/all_datasets/ETT-small/ETTh2.jsonl"

# Read CSV
df = pd.read_csv(in_path)

# Drop/ignore the timestamp column (change name if yours differs)
if "date" in df.columns:
    df = df.drop(columns=["date"])


# (Optional) handle NaNs — here we drop rows that have any NaN
df = df.dropna(axis=0, how="any")

# Write JSONL as one sequence per row
with open(out_path, "w") as f:
    for _, row in df.iterrows():
        rec = {"sequence": row.tolist()}
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {len(df)} lines to {out_path}")
