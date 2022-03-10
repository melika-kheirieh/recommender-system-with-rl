
import torch
import numpy as np


class Env():
    def __init__(self, user_item_matrix):
        self.matrix = user_item_matrix
        self.item_count = item_num
        self.memory = np.ones([user_num, params['N']]) * item_num
        # memory is initialized as [item_num] * N for each user
        # it is padding indexes in state_repr and will result in zero embeddings

    def reset(self, user_id):
        self.user_id = user_id
        self.viewed_items = []
        self.related_items = np.argwhere(self.matrix[self.user_id] > 0)[:, 1]
        self.num_rele = len(self.related_items)
        self.nonrelated_items = np.random.choice(
            list(set(range(self.item_count)) - set(self.related_items)), self.num_rele)
        self.available_items = np.zeros(self.num_rele * 2)
        self.available_items[::2] = self.related_items
        self.available_items[1::2] = self.nonrelated_items

        return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :])

    def step(self, action, action_emb=None, buffer=None):
        initial_user = self.user_id
        initial_memory = self.memory[[initial_user], :]

        reward = float(to_np(action)[0] in self.related_items)
        self.viewed_items.append(to_np(action)[0])
        if reward:
            if len(action) == 1:
                self.memory[self.user_id] = list(self.memory[self.user_id][1:]) + [action]
            else:
                self.memory[self.user_id] = list(self.memory[self.user_id][1:]) + [action[0]]

        if len(self.viewed_items) == len(self.related_items):
            done = 1
        else:
            done = 0

        if buffer is not None:
            buffer.push(np.array([initial_user]), np.array(initial_memory), to_np(action_emb)[0],
                        np.array([reward]), np.array([self.user_id]), self.memory[[self.user_id], :],
                        np.array([reward]))

        return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :]), reward, done


class Actor_DRR(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state):
        return self.layers(state)

    def get_action(self, user, memory, state_repr,
                   action_emb,
                   items=torch.tensor([i for i in range(item_num)]),
                   return_scores=False
                   ):
        state = state_repr(user, memory)
        scores = torch.bmm(ft_item_embeddings_tensor[items, :].double().unsqueeze(0),
                           action_emb.T.unsqueeze(0).double()).squeeze(0)
        if return_scores:
            return scores, torch.gather(items, 0, scores.argmax(0))
        else:
            return torch.gather(items, 0, scores.argmax(0))


class Critic_DRR(nn.Module):
    def __init__(self, state_repr_dim, action_emb_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_repr_dim + action_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.layers(x)
        return x


class State_Repr_Module(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, hidden_dim):
        super().__init__()
        
        DATA_FOLDER = '/content/drive/MyDrive/MIND-Dataset/1/'
        news=pd.read_csv(os.path.join(DATA_FOLDER, 'news_feats'),sep=',')
        users=pd.read_csv(os.path.join(DATA_FOLDER, 'users_feats'),sep=',')

        users.drop(columns='user_id_encoded',inplace=True)
        user_embedding = users.values
        ft_user_embedding_tensor = torch.from_numpy(user_embedding)
        
        ft_item_embeddings_tensor = torch.from_numpy(item_embeddings).double()
        
        self.drr_ave = torch.nn.Conv1d(in_channels=params['N'], out_channels=1, kernel_size=1).double()

        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        user_embedding = ft_user_embedding_tensor[user.long(), :]
        item_embeddings = ft_item_embeddings_tensor[memory.long(), :]
        item_embeddings = item_embeddings.type(torch.DoubleTensor)
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)

        return torch.cat((user_embedding, user_embedding * drr_ave, drr_ave), 1)


def ddpg_update(training_env,
                step=0,
                batch_size=params['batch_size'],
                gamma=params['gamma'],
                min_value=params['min_value'],
                max_value=params['max_value'],
                soft_tau=params['soft_tau'],
               ):
    beta = get_beta(step)
    user, memory, action, reward, next_user, next_memory, done = replay_buffer.sample(batch_size, beta)
    user        = torch.FloatTensor(user)
    memory      = torch.FloatTensor(memory)
    action      = torch.FloatTensor(action)
    reward      = torch.FloatTensor(reward)
    next_user   = torch.FloatTensor(next_user)
    next_memory = torch.FloatTensor(next_memory)
    done = torch.FloatTensor(done)

    state       = state_repr(user, memory)
    policy_loss = value_net(state.float(), policy_net(state.float()))
    policy_loss = -policy_loss.mean()

    next_state     = state_repr(next_user, next_memory)
    next_action    = target_policy_net(next_state.float())
    target_value   = target_value_net(next_state.float(), next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state.float(), action)
    value_loss = value_criterion(value, expected_value.detach())

    state_repr_optimizer.zero_grad()
    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    value_optimizer.step()
    state_repr_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    writer.add_histogram('value', value, step)
    writer.add_histogram('target_value', target_value, step)
    writer.add_histogram('expected_value', expected_value, step)
    writer.add_histogram('policy_loss', policy_loss, step)
