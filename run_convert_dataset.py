import os
import pandas as pd
import json

data_dir = "/scratch/s223540177/Time-Series-Library/data/all_datasets/"
# data_dir = "/scratch/s225250685/project/ICL-time-series/Time-MoE/data/all_datasets"
dir_map = {
    'ETTh1': 'ETT-small/ETTh1.csv',
    'ETTm1': 'ETT-small/ETTm1.csv',
    'ETTh2': 'ETT-small/ETTh2.csv',
    'ETTm2': 'ETT-small/ETTm2.csv',
    'electricity': 'electricity/electricity.csv',
    'exchange': 'exchange_rate/exchange_rate.csv',
    'illness': 'illness/national_illness.csv',
    'traffic': 'traffic/traffic.csv',
    'weather': 'weather/weather.csv'
}


        

for k, v in dir_map.items():
    dir_map[k] = os.path.join(data_dir, v)

dataset = 'ETTh2'
in_path = dir_map[dataset]
out_path = in_path[:-4] + '100.jsonl'

if dataset == 'ETTh1' or dataset == 'ETTh2':
    border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
elif dataset == 'ETTm1' or dataset == 'ETTm2':
    border1s = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
else:
    border1s = None
    border2s = None

# in_path  = "/scratch/s223540177/Time-Series-Library/data/all_datasets/illness/national_illness.csv"
# out_path = "/scratch/s223540177/Time-Series-Library/data/all_datasets/illness/national_illness.jsonl"

# if 

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
        N = len(seq)

        if border1s is not None:
            seq = seq[border1s[0]:border1s[-1]]
        else:
            n_test = int(N*0.2)
            seq = seq[:-n_test]
        
        # only take half of seq for training
        # half_len = len(seq) // 2

        # take 100 examples only
        seq = seq[-100:]
        rec = {"sequence": seq}
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {df.shape[1]} lines to {out_path}")
