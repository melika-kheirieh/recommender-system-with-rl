
from collections import defaultdict
import os
import pickle
import random
import requests
import time
import tqdm

from IPython.core.debugger import set_trace
import numpy as np
import pandas as pd
# from ranger import Ranger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

from utils import (EvalDataset, OUNoise, Prioritized_Buffer, get_beta,
                   preprocess_data, to_np, hit_metric, dcg_metric)

def run_evaluation(net, state_representation, training_env_memory, loader=valid_loader):
    hits = []
    dcgs = []
    test_env = Env(test_matrix)
    test_env.memory = training_env_memory.copy()

    user, memory = test_env.reset(int(to_np(next(iter(valid_loader))['user'])[0]))
    for batch in loader:
        action_emb = net(state_repr(user, memory).float())
        scores, action = net.get_action(
            batch['user'],
            torch.tensor(test_env.memory[to_np(batch['user']).astype(int), :]),
            state_representation,
            action_emb,
            batch['item'].long(),
            return_scores=True
        )
        user, memory, reward, done = test_env.step(action)

        _, ind = scores[:, 0].topk(10)
        predictions = torch.take(batch['item'], ind).cpu().numpy().tolist()
        actual = batch['item'][0].item()
        hits.append(hit_metric(predictions, actual))
        dcgs.append(dcg_metric(predictions, actual))

    return np.mean(hits), np.mean(dcgs)


if __name__ == '__main__':
    valid_dataset = EvalDataset(
    np.array(test_data)[np.array(test_data)[:, 0] == 49107],
    item_num,
    test_matrix)
    valid_loader = td.DataLoader(valid_dataset, batch_size=100, shuffle=False)

    full_dataset = EvalDataset(np.array(test_data), item_num, test_matrix)
    full_loader = td.DataLoader(full_dataset, batch_size=100, shuffle=False)
