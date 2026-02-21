from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import os

# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参数设置
GAMMA = 0.99  # 折扣因子
TAU = 0.005   # 软更新参数
LR = 0.0003   # 降低了学习率以提高稳定性 (原 0.001)
BATCH_SIZE = 256
MEMORY_CAPACITY = 100000

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
 
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
 
    def __len__(self):
        return len(self.buffer)

# Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
 
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
 
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std
 
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # 计算对数概率并进行 Tanh 修正
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

# SAC 智能体
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = device
        self.action_dim = action_dim
 
        # 初始化网络
        self.actor = PolicyNetwork(state_dim, action_dim, max_action).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
 
        # 自动熵调整 (Automatic Entropy Tuning)
        self.target_entropy = -action_dim  # 目标熵设为 -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)
 
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LR)
 
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.max_action = max_action
 
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()[0]
 
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
 
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
 
        # 获取当前 alpha
        alpha = self.log_alpha.exp()

        # 更新 Q 网络
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            q_target = rewards + GAMMA * (1 - dones) * target_q
 
        # 计算 Q 损失并进行梯度裁剪
        curr_q1 = self.q1(states, actions)
        curr_q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(curr_q1, q_target)
        q2_loss = F.mse_loss(curr_q2, q_target)
 
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0) # 梯度裁剪
        self.q1_optimizer.step()
 
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0) # 梯度裁剪
        self.q2_optimizer.step()
 
        # 更新策略网络
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
 
        actor_loss = (alpha * log_probs - q_new).mean()
 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) # 梯度裁剪
        self.actor_optimizer.step()

        # 更新 Alpha [自动熵调整核心]
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
 
    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, path):
        """保存模型权重"""
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'log_alpha': self.log_alpha
        }, path)

    def load(self, path):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'].data)