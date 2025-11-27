import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d
    
    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super(MLP, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=7.5e-4, gamma=0.99, device=None, hidden_dims=(256, 256)):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.qnet = MLP(state_dim, action_dim, hidden_dims).to(self.device)
        self.target = MLP(state_dim, action_dim, hidden_dims).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = ReplayBuffer()
        self.eps = 1.0
        self.eps_min = 0.1
        self.eps_decay = 0.998
    
    def select_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.qnet(s)
        return int(q.argmax().cpu().numpy())
    
    def push(self, *args):
        self.replay.push(*args)
    
    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return 0.0
        s, a, r, ns, d = self.replay.sample(batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        q_values = self.qnet(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(ns).max(1)[0].unsqueeze(1)
        q_target = r + (1.0 - d) * self.gamma * q_next
        
        loss = nn.functional.smooth_l1_loss(q_values, q_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.opt.step()

        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return loss.item()
    
    def sync_target(self):
        self.target.load_state_dict(self.qnet.state_dict())
    
    def save(self, path):
        torch.save({'qnet': self.qnet.state_dict(), 'target': self.target.state_dict()}, path)
    
    def load(self, path):
        d = torch.load(path)
        self.qnet.load_state_dict(d['qnet'])
        self.target.load_state_dict(d['target'])
