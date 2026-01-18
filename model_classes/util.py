import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pandas
import wandb

import torch.nn.functional as F
from model_classes.models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, TemporalFusionTransformer, MambaSimple, TimeBridge



model_dict = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Nonstationary_Transformer': Nonstationary_Transformer,
    'DLinear': DLinear,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'LightTS': LightTS,
    'Reformer': Reformer,
    'ETSformer': ETSformer,
    'PatchTST': PatchTST,
    'Pyraformer': Pyraformer,
    'MICN': MICN,
    'Crossformer': Crossformer,
    'FiLM': FiLM,
    'iTransformer': iTransformer,
    'Koopa': Koopa,
    'TiDE': TiDE,
    'FreTS': FreTS,
    'Mamba': MambaSimple,
    'TimeMixer': TimeMixer,
    'TSMixer': TSMixer,
    'SegRNN': SegRNN,
    'TemporalFusionTransformer': TemporalFusionTransformer,
    'TimeBridge': TimeBridge
}


def build_model(args):
    model = model_dict[args.model].Model(args).float()

    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    return model
