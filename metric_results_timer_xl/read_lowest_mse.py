import os
import re

folder = "/home/s223540177/Time-MoE/metric_results_timer_xl/96_24/illness/final_metrics"

best_file = None
best_mse = float("inf")
nan_count = 0

for fname in os.listdir(folder):
    if fname.endswith(".txt"):
        fpath = os.path.join(folder, fname)

        # Read file content
        with open(fpath, "r") as f:
            text = f.read()

        # Extract mse: <number>
        match = re.search(r"mse:\s*([0-9\.eE+-]+)", text)
        mae = re.search(r"mae:\s*([0-9\.eE+-]+)", text)

        nan_count = re.search(r"nan_count:\s*([0-9]+)", text)
        if match:
            mse = float(match.group(1))
            
            # if nan_count is not None:
            #     nan_count = int(nan_count.group(1))  # now an integer
            #     print(f"File: {fname}, MSE: {mse}, NaN Count: {nan_count}")
            if mse < best_mse:
                best_mse = mse
                best_file = fname
                best_mae = mae
            

# Output result
print("Best file:", best_file)
print("Lowest MSE:", best_mse)
