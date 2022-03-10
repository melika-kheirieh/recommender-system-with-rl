
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



#data_dir = "data"
data_dir = '/content/drive/MyDrive/MIND-Dataset/1/'
#rating = "ml-1m.train.rating"
rating = 'data_encoded'

params = {
    'batch_size': 512,
    'embedding_dim': 12,
    'hidden_dim': 16,
    'N': 5, # memory size for state_repr
    'ou_noise':False,

    'value_lr': 1e-5,
    'value_decay': 1e-4,
    'policy_lr': 1e-5,
    'policy_decay': 1e-6,
    'state_repr_lr': 1e-5,
    'state_repr_decay': 1e-3,
    'log_dir': 'logs/final/',
    'gamma': 0.8,
    'min_value': -10,
    'max_value': 10,
    'soft_tau': 1e-3,

    'buffer_size': 1000000
}

if __name__ == '__main__':
    DATA_FOLDER = '/content/drive/MyDrive/MIND-Dataset/1/'
    news=pd.read_csv(os.path.join(DATA_FOLDER, 'news_feats'),sep=',')
    users=pd.read_csv(os.path.join(DATA_FOLDER, 'users_feats'),sep=',')

    #news['category'] = news.category.astype("category")
    #category_dict = dict(enumerate(news.category.cat.categories))
    #news['cat_encoded'] = news['category'].cat.codes
    #category_dict = {v:k for k,v in category_dict.items()}
    #users = users.replace({'fav_cat_history': category_dict , 'second_fav_cat_history' : category_dict})
    #news['sub_category'] = news.sub_category.astype("category")
    #news['sub_cat_encoded'] = news.sub_category.astype("category").cat.codes

    #def min_max_scale(col):
    #  col = (col - col.min())/(col.max()-col.min())
    #  return col

    #news['cat_encoded'] = min_max_scale(news['cat_encoded'])
    #news['sub_cat_encoded'] = min_max_scale(news['sub_cat_encoded'])
    #users['fav_cat_history'] = min_max_scale(users['fav_cat_history'])
    #users['history_cat_diversity'] = min_max_scale(users['history_cat_diversity'])


    # Movielens (1M) data from the https://github.com/hexiangnan/neural_collaborative_filtering
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    file_path = os.path.join(data_dir, rating)
    if os.path.exists(file_path):
        print("Skip loading " + file_path)
    else:
        with open(file_path, "wb") as tf:
            print("Load " + file_path)
            r = requests.get("https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data/" + rating)
            tf.write(r.content)

    (train_data, train_matrix, test_data, test_matrix,
     user_num, item_num, appropriate_users) = preprocess_data(data_dir, rating)
    ####################################################################################

    ou_noise = OUNoise(params['embedding_dim'], decay_period=10)
    ####################################################################################

    torch.manual_seed(2)

    state_repr = State_Repr_Module(user_num, item_num, params['embedding_dim'], params['hidden_dim'])
    policy_net = Actor_DRR(params['embedding_dim'], params['hidden_dim'])
    value_net  = Critic_DRR(params['embedding_dim'] * 3, params['embedding_dim'], params['hidden_dim'])
    replay_buffer = Prioritized_Buffer(params['buffer_size'])

    target_value_net  = Critic_DRR(params['embedding_dim'] * 3, params['embedding_dim'], params['hidden_dim'])
    target_policy_net = Actor_DRR(params['embedding_dim'], params['hidden_dim'])

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    value_criterion  = nn.MSELoss()
    value_optimizer  = Ranger(value_net.parameters(),  lr=params['value_lr'],
    # value_optimizer  = torch.optim.Adam(value_net.parameters(),  lr=params['value_lr'],
                              weight_decay=params['value_decay'])
    policy_optimizer = Ranger(policy_net.parameters(), lr=params['policy_lr'],
                              weight_decay=params['policy_decay'])
    state_repr_optimizer = Ranger(state_repr.parameters(), lr=params['state_repr_lr'],
                                  weight_decay=params['state_repr_decay'])

    writer = SummaryWriter(log_dir=params['log_dir'])
    ####################################################################################

    np.random.seed(16)
    train_env = Env(train_matrix)
    hits, dcgs = [], []
    hits_all, dcgs_all = [], []
    step, best_step = 0, 0
    step, best_step, best_step_all = 0, 0, 0
    users = np.random.permutation(appropriate_users)
    ou_noise = OUNoise(params['embedding_dim'], decay_period=10)

    for u in tqdm.tqdm(users):
        user, memory = train_env.reset(u)
        if params['ou_noise']:
            ou_noise.reset()
        for t in range(int(train_matrix[u].sum())):
            action_emb = policy_net(state_repr(user, memory).float())
            if params['ou_noise']:
                action_emb = ou_noise.get_action(action_emb.detach().cpu().numpy()[0], t)
            action = policy_net.get_action(
                user,
                torch.tensor(train_env.memory[to_np(user).astype(int), :]),
                state_repr,
                action_emb,
                torch.tensor(
                    [item for item in train_env.available_items
                    if item not in train_env.viewed_items]
                ).long()
            )
            user, memory, reward, done = train_env.step(
                action,
                action_emb,
                buffer=replay_buffer
            )

            if len(replay_buffer) > params['batch_size']:
                ddpg_update(train_env, step=step)

            if step % 100 == 0 and step > 0:
                hit, dcg = run_evaluation(policy_net, state_repr, train_env.memory)
                writer.add_scalar('hit', hit, step)
                writer.add_scalar('dcg', dcg, step)
                hits.append(hit)
                dcgs.append(dcg)
                if np.mean(np.array([hit, dcg]) - np.array([hits[best_step], dcgs[best_step]])) > 0:
                    best_step = step // 100
                    torch.save(policy_net.state_dict(), save_path + 'policy_net.pth')
                    torch.save(value_net.state_dict(), save_path + 'value_net.pth')
                    torch.save(state_repr.state_dict(), save_path + 'state_repr.pth')
            if step % 1000 == 0 and step > 0:
                hit, dcg = run_evaluation(policy_net, state_repr, train_env.memory, full_loader)
                writer.add_scalar('hit_all', hit, step)
                writer.add_scalar('dcg_all', dcg, step)
                hits_all.append(hit)
                dcgs_all.append(dcg)
                if np.mean(np.array([hit, dcg]) - np.array([hits_all[best_step_all], dcgs_all[best_step_all]])) > 0:
                    best_step_all = step // 10000
                    torch.save(policy_net.state_dict(), save_path + 'best_policy_net.pth')
                    torch.save(value_net.state_dict(), save_path + 'best_value_net.pth')
                    torch.save(state_repr.state_dict(), save_path + 'best_state_repr.pth')
            step += 1
    ####################################################################################

    
