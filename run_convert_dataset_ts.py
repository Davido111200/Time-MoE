from datetime import datetime
from distutils.util import strtobool

import pandas as pd
import os
import json

data_dir = "/scratch/s223540177/Time-Series-Library/data/all_datasets/"
# data_dir = "/scratch/s225250685/project/ICL-time-series/Time-MoE/data/all_datasets"
dir_map = {
    'solar': 'solar_10_minutes_dataset.tsf',
    'm4_yearly': 'm4_yearly_dataset.tsf',
    'tourism_monthly': 'tourism_monthly_dataset.tsf',
    'tourism_yearly': 'tourism_yearly_dataset.tsf',
    'tourism_quarterly': 'tourism_quarterly_dataset.tsf',
    'us_births': 'us_births_dataset.tsf',
    'hospital': 'hospital_dataset.tsf',
    'rideshare': 'rideshare_dataset_without_missing_values.tsf',
    'nn5': 'nn5_daily_dataset_without_missing_values.tsf',
    'pedestrian': 'pedestrian_counts_dataset.tsf',
    'temperature': 'temperature_rain_dataset_without_missing_values.tsf',
    'fredmd': 'fred_md_dataset.tsf',
    'saugeenday': 'saugeenday_dataset.tsf',
    'sunspots': 'sunspot_dataset_without_missing_values.tsf',
    'bitcoin': 'bitcoin_dataset_without_missing_values.tsf',
    'kdd': 'kdd_cup_2018_dataset_without_missing_values.tsf',
    'vehicle': 'vehicle_trips_dataset_without_missing_values.tsf',
    'm1_monthly': 'm1_monthly_dataset.tsf',

}

for k, v in dir_map.items():
    dir_map[k] = os.path.join(data_dir, v)


dataset = 'tourism_quarterly'  # change dataset here
in_path = dir_map[dataset]
out_path = in_path[:-4] + '.jsonl'




# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


# Example of usage
loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(in_path)

print(loaded_data)
print(frequency)
print(forecast_horizon)
print(contain_missing_values)
print(contain_equal_length)


value_column_name = "series_value"

with open(out_path, "w") as f:
    for _, row in loaded_data.iterrows():
        # Convert the pandas array to a plain Python list
        series = list(row[value_column_name])
        N = len(series)

        # match your CSV logic: drop last 20% as "test"
        n_test = int(N * 0.2)
        seq = series[:-n_test] if n_test > 0 else series

        # (minimal version, like your illness script)
        rec = {"sequence": seq}
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {len(loaded_data)} lines to {out_path}")


value_column_name = "series_value"

# Map TSF frequency string to a pandas offset alias
freq_map = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "weekly": "W",
    "daily": "D",
    "hourly": "H",
}
pd_freq = freq_map.get(str(frequency).lower(), "D")  # default to daily

wide_frames = []

for _, row in loaded_data.iterrows():
    series_name = row["series_name"]          # e.g. T1, T2, or whatever is in the TSF
    start = row["start_timestamp"]
    series = list(row[value_column_name])

    # Replace "NaN" placeholder with real NaN so CSV looks clean
    clean_series = [float(x) if x != "NaN" else float("nan") for x in series]

    # Build date index for this series
    dates = pd.date_range(start=start, periods=len(clean_series), freq=pd_freq)

    tmp = pd.DataFrame({
        "date": dates,
        series_name: clean_series,
    })
    wide_frames.append(tmp)

# Merge all series on 'date'
wide_df = wide_frames[0]
for tmp in wide_frames[1:]:
    wide_df = wide_df.merge(tmp, on="date", how="outer")

# Sort and format date as "YYYY-MM-DD HH:MM:SS"
wide_df = wide_df.sort_values("date").reset_index(drop=True)
wide_df["date"] = wide_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

csv_out_path = in_path[:-4] + ".csv"
wide_df.to_csv(csv_out_path, index=False)
print(f"Wrote wide CSV to {csv_out_path}")
