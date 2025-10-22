import pandas as pd
import json

in_path  = "/scratch/s223540177/Time-Series-Library/data/all_datasets/illness/national_illness.csv"
out_path = "/scratch/s223540177/Time-Series-Library/data/all_datasets/illness/national_illness.jsonl"

# Read CSV
df = pd.read_csv(in_path)

# Drop/ignore the timestamp column (change name if yours differs)
if "date" in df.columns:
    df = df.drop(columns=["date"])

# (Optional) handle NaNs — here we drop rows that have any NaN
df = df.dropna(axis=0, how="any")

# Write JSONL as one sequence per COLUMN (first column sequence, then second, etc.)
with open(out_path, "w") as f:
    for col in df.columns:
        seq = df[col].tolist()
        rec = {"sequence": seq}
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {df.shape[1]} lines to {out_path}")
