from typing import List, Union
from pathlib import Path
import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowWriter
from chronos import BaseChronosPipeline, Chronos2Pipeline
from pathlib import Path
import shutil
# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
)

PATH = "/scratch/s223540177/Time-Series-Library/data/all_datasets"
SAVE_DIR = "/scratch/s223540177/Time-MoE/chronos"


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length), where each row is
    interpreted as one independent time series.
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    )

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")

    # time_series: list of 1D arrays OR 2D array [num_series, time_length]
    if isinstance(time_series, np.ndarray):
        series_list = [time_series[i] for i in range(time_series.shape[0])]
    else:
        series_list = time_series

    dataset = [{"start": start, "target": ts} for ts in series_list]

    ArrowWriter(compression=compression).write_to_file(dataset, path=path)


# ------------------ ETT dataset selection ------------------ #

dataset = "ETTm2"  # options: ETTh1, ETTh2, ETTm1, ETTm2

dataset_to_data_path = {
    "ETTh1": f"{PATH}/ETT-small/ETTh1.csv",
    "ETTh2": f"{PATH}/ETT-small/ETTh2.csv",
    "ETTm1": f"{PATH}/ETT-small/ETTm1.csv",
    "ETTm2": f"{PATH}/ETT-small/ETTm2.csv",
}

data_path = dataset_to_data_path[dataset]

if dataset in ["ETTh1", "ETTh2"]:
    border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
elif dataset in ["ETTm1", "ETTm2"]:
    border1s = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
    border2s = [
        12 * 30 * 24 * 4,
        12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
        12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
    ]
else:
    raise ValueError(f"Unknown dataset: {dataset}")

# ------------------ Load ETT and split ------------------ #

df = pd.read_csv(data_path)

# Drop timestamp column; Chronos will just see numeric arrays for fine-tuning
if "date" in df.columns:
    df = df.drop(columns=["date"])

# Optional: handle NaNs
df = df.dropna(axis=0, how="any").reset_index(drop=True)

# Standard ETT splits (train / val / test)
train_df = df.iloc[border1s[0] : border2s[0]]
val_df   = df.iloc[border1s[1] : border2s[1]]
test_df  = df.iloc[border1s[2] : border2s[2]]

print("Shapes: train", train_df.shape, "val", val_df.shape, "test", test_df.shape)

# ------------------ Chronos-2 fine-tuning inputs ------------------ #
# MANY INDEPENDENT SERIES: EACH COLUMN IS ITS OWN TARGET

target_cols = df.columns.tolist()  # all remaining columns are targets
print("Target columns:", target_cols)


def make_chronos_inputs(frame: pd.DataFrame):
    """
    Create Chronos inputs where each column is treated as an independent
    univariate time series.

    Returns:
        List[dict], one dict per column:
            {
                "target": np.ndarray of shape (T,),
                # no covariates
            }
    """
    inputs = []
    for col in target_cols:
        ts = frame[col].to_numpy(dtype="float32")  # shape (T,)
        inputs.append(
            {
                "target": ts,
                # IMPORTANT: do NOT pass None for covariates; just omit them
                # "past_covariates": None,
                # "future_covariates": None,
            }
        )
    return inputs


train_inputs = make_chronos_inputs(train_df)
val_inputs   = make_chronos_inputs(val_df)

print("Number of series in train:", len(train_inputs))
print("Length of each series (train, first col):", train_inputs[0]["target"].shape)

# Fine-tune Chronos-2 on all columns as independent series
finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    validation_inputs=val_inputs,  # optional, but good to have
    prediction_length=192,  # or whatever horizon you want
    num_steps=100,
    learning_rate=1e-5,
    batch_size=32,
    logging_steps=10,
    save_steps=1000
)

# ------------------ Arrow export (one series per column) ------------------ #

# Export all columns as separate univariate series
train_series = train_df[target_cols].to_numpy(dtype="float32").T  # (num_series, T_train)
val_series   = val_df[target_cols].to_numpy(dtype="float32").T
test_series  = test_df[target_cols].to_numpy(dtype="float32").T

out_dir = Path(PATH) / "ETT-small" / "chronos_arrow" / dataset
out_dir.mkdir(parents=True, exist_ok=True)

convert_to_arrow(out_dir / "train.arrow", train_series)
convert_to_arrow(out_dir / "val.arrow",   val_series)
convert_to_arrow(out_dir / "test.arrow",  test_series)

print("Saved Arrow files to:", out_dir)

# Save the trained model
save_path = Path(SAVE_DIR) / f"{dataset}_finetuned"
save_path.parent.mkdir(parents=True, exist_ok=True)
finetuned_pipeline.save_pretrained(str(save_path))
print("Saved finetuned Chronos-2 model to:", save_path)

# Delete this folder: /home/s223540177/Time-MoE/chronos-2-finetuned
ckpt_dir = Path("/home/s223540177/Time-MoE/chronos-2-finetuned")

if ckpt_dir.exists() and ckpt_dir.is_dir():
    shutil.rmtree(ckpt_dir)
    print(f"Deleted: {ckpt_dir}")
else:
    print(f"Path does not exist or is not a directory: {ckpt_dir}")
