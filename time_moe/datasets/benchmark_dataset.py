#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np
import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader

from time_moe.datasets.general_dataset import GeneralDataset
from time_moe.utils.log_util import log_in_local_rank_0

def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


class BenchmarkEvalDataset(Dataset):

    def __init__(self, csv_path, seq_len: int, pred_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)

        base_name = os.path.basename(csv_path).lower()
        if 'etth' in base_name:
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ettm' in base_name:
            border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
            border2s = [num_train, num_train + num_vali, len(df)]

        
        # start_dt = df.iloc[border1s[2]]['date']
        # eval_start_dt = df.iloc[border1s[2] + seq_len]['date']
        # end_dt = df.iloc[border2s[2] - 1]['date']
        # log_in_local_rank_0(f'>>> Split test data from {start_dt} to {end_dt}, '
        #                     f'and evaluation start date is: {eval_start_dt}')

        if 'm4' in csv_path.lower():
            # df_values = df.series_value
            df["series_value"] = df["series_value"].apply(json.loads)
            df_values = df['series_value']
        else:
            cols = df.columns[1:]
            df_values = df[cols].values
        

        train_data = df_values[border1s[0]:border2s[0]]
        test_data = df_values[border1s[2]:border2s[2]]
        # scaling
        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_test_data = scaler.transform(test_data)

        # assignment
        self.hf_dataset = scaled_test_data.transpose(1, 0)
        self.num_sequences = len(self.hf_dataset)
        # 1 for the label
        self.window_length = self.seq_len + self.pred_len


        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]
        

        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)

        return {
            'inputs': np.array(window_seq[: self.seq_len], dtype=np.float32),
            'labels': np.array(window_seq[-self.pred_len:], dtype=np.float32),
            'channel_id': np.array(seq_i),
        }



class BenchmarkEvalDatasetTrain(Dataset): 
    def __init__(self, args, csv_path, seq_len: int, pred_len: int, max_train_samples=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.args = args
        self.label_len = args.label_len

        df = pd.read_csv(csv_path)
        

        base_name = os.path.basename(csv_path).lower()
        if 'etth' in base_name:
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ettm' in base_name:
            border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
            border2s = [num_train, num_train + num_vali, len(df)]

        df_stamp = df[['date']].iloc[border1s[0]:border2s[0]].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        # month / day / weekday / hour
        data_stamp = pd.DataFrame({
            'month':   df_stamp['date'].dt.month,
            'day':     df_stamp['date'].dt.day,
            'weekday': df_stamp['date'].dt.weekday,
            'hour':    df_stamp['date'].dt.hour,
        }).values.astype(np.float32)

        self.data_stamp = data_stamp                          # shape [T_train, 4]

        cols = df.columns[1:]
        df_values = df[cols].values

        train_data = df_values[border1s[0]:border2s[0]]
        if max_train_samples is not None:
            train_data = train_data[-max_train_samples:, :]


        # # TODO: DOUBLE CHECK THE SCALE PART
        # if max_train_samples is not None:
        #     if max_train_samples > len(train_data):
        #         pass
        #     else:
        #         # take the last max_train_samples from training set
        #         train_data = train_data[-max_train_samples:]

        # print(f"Train data shape: {train_data.shape}")

        # print(f"Final train data shape: {train_data.shape}")

        # train_data = df_values[0:800]
        # test_data = df_values[800:1600]

        # scaling
        scaler = StandardScaler()
        scaler.fit(train_data)
        # scaled_test_data = scaler.transform(test_data)
        scaled_train_data = scaler.transform(train_data)

        # assignment
        # NOTE: fix here
        self.hf_dataset = scaled_train_data.transpose(1, 0)
        self.num_sequences = len(self.hf_dataset)
        # 1 for the label
        self.window_length = self.seq_len + self.pred_len


        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            print("N points:", n_points)
            print("Window length:", self.window_length)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

        if torch.cuda.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset=self.sub_seq_indexes, shuffle=False)
        else:
            sampler = None

        self.data_loader = DataLoader(
            self.sub_seq_indexes,                     # list of (seq_idx, offset_idx)
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=sampler,
            prefetch_factor=2,
            drop_last=False,
            collate_fn=self._collate_like_ett_hour,    # <— returns EXACT ETT tuple
        )
        # -------------------------------------------------------

    def _collate_like_ett_hour(self, batch_idx_tuples):
        """
        Returns EXACTLY the ETT tuple:
        seq_x      : [B, seq_len, 1]
        seq_y      : [B, label_len + pred_len, 1]
        seq_x_mark : [B, seq_len, 4]            (month, day, weekday, hour)
        seq_y_mark : [B, label_len + pred_len, 4]
        Indexing follows Dataset_ETT_hour semantics.
        """

        B = len(batch_idx_tuples)
        Lx = self.seq_len
        Ld = self.label_len
        Ly = self.pred_len
        Dm = self.data_stamp.shape[1]  # 4

        # allocate
        seq_x      = np.empty((B, Lx, 1), dtype=np.float32)
        seq_y      = np.empty((B, Ld + Ly, 1), dtype=np.float32)
        seq_x_mark = np.empty((B, Lx, Dm), dtype=np.float32)
        seq_y_mark = np.empty((B, Ld + Ly, Dm), dtype=np.float32)

        # NOTE: self.hf_dataset is [C, T_train]; each tuple carries (seq_idx, offset_idx)
        window_len = Lx + Ly
        for i, (seq_i, offset_i) in enumerate(batch_idx_tuples):
            # window is [offset - (seq_len+pred_len), offset) on the TRAIN time axis
            left = offset_i - window_len
            s_begin = left
            s_end   = s_begin + Lx
            r_begin = s_end - Ld
            r_end   = r_begin + Ld + Ly

            # series for this channel
            series = self.hf_dataset[seq_i]              # [T_train]
            window = np.asarray(series[left:offset_i], dtype=np.float32)  # [seq_len + pred_len]

            # EXACT ETT slicing
            seq_x[i, :, 0] = window[:Lx]
            seq_y[i, :, 0] = window[Lx - Ld :]

            # time marks pulled from the SAME train time indices (global over time)
            seq_x_mark[i] = self.data_stamp[s_begin:s_end]
            seq_y_mark[i] = self.data_stamp[r_begin:r_end]

        # return EXACT tuple order as in Dataset_ETT_hour.__getitem__
        return (
            torch.from_numpy(seq_x),      # seq_x
            torch.from_numpy(seq_y),      # seq_y
            torch.from_numpy(seq_x_mark), # seq_x_mark
            torch.from_numpy(seq_y_mark), # seq_y_mark
        )


    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]
        

        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)

        return {
            'inputs': np.array(window_seq[: self.seq_len], dtype=np.float32),
            'labels': np.array(window_seq[-self.pred_len:], dtype=np.float32),
            'channel_id': np.array(seq_i),
        }


class BenchmarkEvalDatasetValid(Dataset):
    def __init__(self, csv_path, seq_len: int, pred_len: int):
        super().__init__()
        import math

        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)

        base_name = os.path.basename(csv_path).lower()
        if 'etth' in base_name:
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ettm' in base_name:
            border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [0, num_train - seq_len, len(df) - num_test - seq_len]
            border2s = [num_train, num_train + num_vali, len(df)]

        # columns (skip timestamp)
        cols = df.columns[1:]
        df_values = df[cols].values

        # train slice (for scaling)
        train_data = df_values[border1s[0]:border2s[0]]

        # ---- keep only last 10% of the validation split (with seq_len history buffer) ----
        last_percent = 0.10
        v1, v2 = border1s[1], border2s[1]  # full valid block [v1, v2)
        valid_len = max(v2 - v1, 0)
        take_last = max(int(math.ceil(last_percent * valid_len)), 1)

        # include history buffer of seq_len for context; clamp to v1
        new_v2 = v2
        new_v1 = max(v1, new_v2 - take_last - self.seq_len)

        # ensure we have enough points for at least one window
        self.window_length = self.seq_len + self.pred_len
        if (new_v2 - new_v1) <= self.window_length:
            new_v1 = max(v1, new_v2 - (self.window_length + 1))

        valid_data = df_values[new_v1:new_v2]

        # logging (optional; uses "valid" wording)
        try:
            start_dt = df.iloc[new_v1]['date']
            eval_start_dt = df.iloc[min(new_v1 + self.seq_len, len(df) - 1)]['date']
            end_dt = df.iloc[new_v2 - 1]['date']
            log_in_local_rank_0(
                f'>>> Split valid data (last 10%) from {start_dt} to {end_dt}, '
                f'and evaluation start date is: {eval_start_dt}'
            )
        except Exception:
            pass

        # scaling: fit on train only, transform valid
        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_valid_data = scaler.transform(valid_data)

        # assignment
        self.hf_dataset = scaled_valid_data.transpose(1, 0)  # [C, T_valid_slice]
        print("HF dataset shape (C, T):", self.hf_dataset.shape)
        self.num_sequences = len(self.hf_dataset)

        # enumerate sub-windows per channel
        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]
        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)
        return {
            'inputs': np.array(window_seq[: self.seq_len], dtype=np.float32),
            'labels': np.array(window_seq[-self.pred_len:], dtype=np.float32),
            'channel_id': np.array(seq_i),
        }


class GeneralEvalDataset(Dataset):

    def __init__(self, data_path, seq_len: int, pred_len: int, onfly_norm: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.onfly_norm = onfly_norm
        self.window_length = self.seq_len + self.pred_len
        self.dataset = GeneralDataset(data_path)
        
        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.dataset[seq_i]

        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)

        inputs = np.array(window_seq[: self.seq_len], dtype=np.float32)
        labels = np.array(window_seq[-self.pred_len:], dtype=np.float32)

        if self.onfly_norm:
            mean_ = inputs.mean()
            std_ = inputs.std()
            if std_ == 0:
                std_ = 1
            inputs = (inputs - mean_) / std_
            labels = (labels - mean_) / std_

        return {
            'inputs': np.array(window_seq[: self.seq_len], dtype=np.float32),
            'labels': np.array(window_seq[-self.pred_len:], dtype=np.float32),
            'channel_id': np.array(seq_i),
        }
