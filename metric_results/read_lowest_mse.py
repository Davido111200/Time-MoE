import os
import re

folder = "/home/s223540177/Time-MoE/metric_results/512_96/saugeenday_dataset/final_metrics"

best_file = None
best_mse = float("inf")

for fname in os.listdir(folder):
    if fname.endswith(".txt"):
        fpath = os.path.join(folder, fname)

        # Read file content
        with open(fpath, "r") as f:
            text = f.read()

        # Extract mse: <number>
        match = re.search(r"mse:\s*([0-9\.eE+-]+)", text)
        mae = re.search(r"mae:\s*([0-9\.eE+-]+)", text)
        if match:
            mse = float(match.group(1))

            if mse < best_mse:
                best_mse = mse
                best_file = fname
                best_mae = mae

# Output result
print("Best file:", best_file)
print("Lowest MSE:", best_mse)
print("Corresponding MAE:", best_mae.group(1) if best_mae else "N/A")
