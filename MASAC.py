import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
 
    def push(self, state_all, action_all, reward, next_state_all, done):
        self.buffer.append((state_all, action_all, reward, next_state_all, done))
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
 
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    # MASAC.py
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        # 必须限制在 [-20, 2] 之间，防止 std 变成 0 或无穷大
        log_std = torch.clamp(log_std, min=-20, max=2) 
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

class CentralizedCritic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super(CentralizedCritic, self).__init__()
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state_all, action_all):
        x = torch.cat([state_all, action_all], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MASACAgent:
    def __init__(self, state_dim, action_dim, total_state_dim, total_action_dim, state_idx, action_idx, lr=0.00005):
        self.state_idx = state_idx  
        self.action_idx = action_idx 
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1 = CentralizedCritic(total_state_dim, total_action_dim).to(device)
        self.q2 = CentralizedCritic(total_state_dim, total_action_dim).to(device)
        self.q1_target = CentralizedCritic(total_state_dim, total_action_dim).to(device)
        self.q2_target = CentralizedCritic(total_state_dim, total_action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor([-2.0], requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.replay_buffer = ReplayBuffer(100000)
        self.gamma, self.tau = 0.99, 0.005

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            mu, _ = self.actor(state)
            return torch.tanh(mu).detach().cpu().numpy()[0]
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, all_agents, batch_size=256):
        if len(self.replay_buffer) < batch_size: return
        s_all, a_all, r, ns_all, done = self.replay_buffer.sample(batch_size)
        s_all, a_all, r, ns_all, done = [torch.FloatTensor(x).to(device) for x in [s_all, a_all, r, ns_all, done]]
        r, done = r.unsqueeze(1), done.unsqueeze(1)

        with torch.no_grad():
            next_actions, curr_next_lp = [], None
            for agent in all_agents:
                ns_local = ns_all[:, agent.state_idx[0]:agent.state_idx[1]]
                na, nlp = agent.actor.sample(ns_local)
                next_actions.append(na)
                if agent == self: curr_next_lp = nlp
            target_q = torch.min(self.q1_target(ns_all, torch.cat(next_actions, dim=1)), 
                                 self.q2_target(ns_all, torch.cat(next_actions, dim=1)))
            y = r + (1 - done) * self.gamma * (target_q - self.log_alpha.exp() * curr_next_lp)
            y = torch.clamp(y, min=-1000.0, max=100.0)

        q1_loss, q2_loss = F.mse_loss(self.q1(s_all, a_all), y), F.mse_loss(self.q2(s_all, a_all), y)
        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 0.5)
        self.q_optimizer.step()

        new_act_i, lp_i = self.actor.sample(s_all[:, self.state_idx[0]:self.state_idx[1]])
        a_all_actor = a_all.clone().detach()
        a_all_actor[:, self.action_idx[0]:self.action_idx[1]] = new_act_i
        q_val = torch.min(self.q1(s_all, a_all_actor), self.q2(s_all, a_all_actor))
        actor_loss = (self.log_alpha.exp() * lp_i - q_val).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (lp_i + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-5.0, 2.0) # 强制限制 alpha 的范围

        for t, p in zip(self.q1_target.parameters(), self.q1.parameters()): t.data.copy_(t.data * (1-self.tau) + p.data * self.tau)
        for t, p in zip(self.q2_target.parameters(), self.q2.parameters()): t.data.copy_(t.data * (1-self.tau) + p.data * self.tau)

    def save(self, path):
        torch.save({'actor': self.actor.state_dict()}, path)