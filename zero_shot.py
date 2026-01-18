from tqdm import tqdm
import numpy as np
import argparse
import os
import torch
from data_provider.data_factory import data_provider
from utils.metrics import metric
from transformers import AutoModelForCausalLM
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# ----------------------
# Reproducibility
# ----------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)

# ----------------------
# Data
# ----------------------
def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

# model define (kept for compatibility with your codebase)
parser.add_argument('--expand', type=int, default=2)
parser.add_argument('--d_conv', type=int, default=4)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--num_kernels', type=int, default=6)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--distil', action='store_false', default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--channel_independence', type=int, default=1)
parser.add_argument('--decomp_method', type=str, default='moving_avg')
parser.add_argument('--use_norm', type=int, default=1)
parser.add_argument('--down_sampling_layers', type=int, default=0)
parser.add_argument('--down_sampling_window', type=int, default=1)
parser.add_argument('--down_sampling_method', type=str, default=None)
parser.add_argument('--seg_len', type=int, default=96)

# optimization
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--use_amp', action='store_true', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_type', type=str, default='cuda')  # cuda or mps
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
parser.add_argument('--p_hidden_layers', type=int, default=2)

# metrics (dtw)
parser.add_argument('--use_dtw', type=bool, default=False)

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--jitter', default=False, action="store_true")
parser.add_argument('--scaling', default=False, action="store_true")
parser.add_argument('--permutation', default=False, action="store_true")
parser.add_argument('--randompermutation', default=False, action="store_true")
parser.add_argument('--magwarp', default=False, action="store_true")
parser.add_argument('--timewarp', default=False, action="store_true")
parser.add_argument('--windowslice', default=False, action="store_true")
parser.add_argument('--windowwarp', default=False, action="store_true")
parser.add_argument('--rotation', default=False, action="store_true")
parser.add_argument('--spawner', default=False, action="store_true")
parser.add_argument('--dtwwarp', default=False, action="store_true")
parser.add_argument('--shapedtwwarp', default=False, action="store_true")
parser.add_argument('--wdba', default=False, action="store_true")
parser.add_argument('--discdtw', default=False, action="store_true")
parser.add_argument('--discsdtw', default=False, action="store_true")
parser.add_argument('--extra_tag', type=str, default="")
# TimeXer
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--weighted_loss', type=int, default=0)

args = parser.parse_args()

# ----------------------
# Load data
# ----------------------
train_data, train_loader = get_data(args, flag='train')
vali_data, vali_loader = get_data(args, flag='val')
test_data, test_loader = get_data(args, flag='test')
print('test_data: ', len(test_data))

# ----------------------
# Load TimeMoE model (zero-shot)
# ----------------------
device = args.gpu_type if (args.use_gpu and torch.cuda.is_available() and args.gpu_type == 'cuda') else 'cpu'
torch.cuda.set_device(args.gpu) if (device == 'cuda') else None

# Try flash-attn; fall back gracefully if unavailable
try:
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map='auto' if device == 'cuda' else device,
        attn_implementation='flash_attention_2',
        trust_remote_code=True,
    )
except Exception:
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map='auto' if device == 'cuda' else device,
        trust_remote_code=True,
    )
model.eval()
torch.set_grad_enabled(False)

# ----------------------
# Inference loop (format identical to your original)
#   - uses the last channel only ([-1]) exactly like before
#   - saves preds/true with the same shapes and filenames
#   - prints per-iter metrics and final metrics
# ----------------------
num_eval_samples = len(test_loader)
preds = []
trues = []

eps = 1e-6
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
    # batch_x: [B, T, D]; original code does transpose + squeeze(0) then select last channel
    batch_x = batch_x.transpose(1, 2).squeeze(0)     # -> [D, T]
    batch_y = batch_y.transpose(1, 2).squeeze(0)     # -> [D, pred_len]

    # strictly follow original behavior: take last channel
    hist_data = batch_x[-1]                           # 1D, length = seq_len
    target_y  = batch_y[-1:]                          # keep channel axis -> shape (1, pred_len)

    # normalize per-sequence (as in your TimeMoE snippet)
    t = torch.as_tensor(hist_data, dtype=torch.float32).unsqueeze(0)  # [1, seq_len]
    mean = t.mean(dim=-1, keepdim=True)
    std  = t.std(dim=-1, keepdim=True)
    std  = torch.where(std < eps, torch.full_like(std, eps), std)
    normed = (t - mean) / std

    # generate
    pred_len = int(args.pred_len)
    # TimeMoE's generate returns [B, seq_len + pred_len]; slice the tail
    out = model.generate(normed.to(model.device), max_new_tokens=pred_len)
    normed_pred = out[:, -pred_len:].detach().to('cpu')  # [1, pred_len]

    # inverse normalize to original scale and move to numpy
    pred = (normed_pred * std + mean).squeeze(0).numpy()  # [pred_len]
    outputs = np.expand_dims(pred, axis=0)                 # [1, pred_len]  (STRICTLY SAME as original)

    trues.append(target_y.numpy())     # [1, pred_len]
    preds.append(outputs)              # [1, pred_len]

    mse = np.mean((outputs - target_y.numpy()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(outputs - target_y.numpy()))
    print(f'Iter: {i}, mse: {mse:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}')

    if i == num_eval_samples:
        break

trues = np.concatenate(trues, axis=0)   # [N, pred_len]
preds = np.concatenate(preds, axis=0)   # [N, pred_len]
mae, mse, rmse, mape, mspe = metric(preds, trues)

folder_path = 'test_results/zeroshot/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
np.save(folder_path + 'pred.npy', preds)
np.save(folder_path + 'true.npy', trues)

print('mse: ', mse, 'rmse: ', rmse, 'mae: ', mae, 'mape: ', mape, 'mspe: ', mspe)
