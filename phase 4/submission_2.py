import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import math

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_SAC_ACTOR = None
_UNWEDGER = None
_FRAME_STACK = deque(maxlen=4)
_UNWEDGE_HIDDEN = None
_UNWEDGE_PREV_ACTION = None
_SAFETY_MODULE = None
x = 0.0
y = 0.0
theta = 0.0
_LAST_RNG = None

class SpatialSafetyModule:
    def __init__(self, grid_res=20):
        self.res = grid_res
        self.flagged_cells = set() 
        
        # Hyperparameters
        self.detection_range = 100.0  
        self.fov_deg = 15.0          
        self.max_penalty = 500.0      

    def flag_stuck(self, x, y):
        x, y = float(x), float(y)

        ix = int(x // self.res)
        iy = int(y // self.res)        
        self.flagged_cells.add((ix, iy))

    def get_forward_penalty(self, x, y, theta):
        x, y, theta = float(x), float(y), float(theta)
        
        heading_vec = np.array([np.cos(theta), np.sin(theta)])
        total_penalty = 0.0
        
        for ix, iy in self.flagged_cells:
            cell_x = (ix * self.res) + (self.res / 2)
            cell_y = (iy * self.res) + (self.res / 2)
            
            to_cell = np.array([cell_x - x, cell_y - y])
            dist = np.linalg.norm(to_cell)
            
            if dist > self.detection_range or dist < 2.0:
                continue
                
            to_cell_unit = to_cell / dist
            dot_product = np.dot(heading_vec, to_cell_unit)
            
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            if angle_deg < self.fov_deg / 2:
                dist_factor = (self.detection_range - dist) / self.detection_range
                alignment_factor = dot_product 
                
                cell_penalty = self.max_penalty * dist_factor * alignment_factor
                total_penalty = max(total_penalty, cell_penalty)

        return total_penalty

    def mask_logits(self, logits, x, y, theta):
        penalty = self.get_forward_penalty(x, y, theta)
        masked_logits = logits.clone()
        masked_logits[0, 2] -= penalty
        return masked_logits
    
    def clear_memory(self):
        self.flagged_cells.clear()

class SACActor(nn.Module):
    def __init__(self, state_dim=72, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.net(state)

class RecurrentPPO(nn.Module):
    def __init__(self, action_dim=5, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.input_layer = nn.Linear(action_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, prev_action, hidden):
        prev_action_onehot = F.one_hot(prev_action, num_classes=self.action_dim).float()
        x = F.relu(self.input_layer(prev_action_onehot))
        x = x.unsqueeze(1) 
        x, next_hidden = self.gru(x, hidden)
        x = x.squeeze(1)
        return self.actor(x), None, next_hidden


def _load_once():
    global _SAC_ACTOR, _UNWEDGER, _SAFETY_MODULE
    if _SAC_ACTOR is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "FSAC_with_gru_500_no_walls.pth")
    checkpoint = torch.load(wpath, map_location="cpu")

    _SAC_ACTOR = SACActor()
    _SAC_ACTOR.load_state_dict(checkpoint['actor_state_dict'])
    _SAC_ACTOR.eval()

    _UNWEDGER = RecurrentPPO()
    _UNWEDGER.load_state_dict(checkpoint['unwedger'])
    _UNWEDGER.eval()

    _SAFETY_MODULE = SpatialSafetyModule(grid_res=25)

def update_positional_data(action, is_stuck):
    global x, y, theta
    theta_deltas = (
            math.pi / 4.0,   # 0: L45
            math.pi / 8.0,   # 1: L22
            0.0,             # 2: FWD
            -math.pi / 8.0,  # 3: R22
            -math.pi / 4.0   # 4: R45
        )
    theta = (theta + theta_deltas[action]) % (2 * math.pi)
    if action == 2 and not is_stuck:
        x += math.cos(theta)
        y += math.sin(theta)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _FRAME_STACK, _UNWEDGE_HIDDEN, _UNWEDGE_PREV_ACTION, _SAFETY_MODULE, x, y, theta, _LAST_RNG
    _load_once()

    if rng is not _LAST_RNG:
        _LAST_RNG = rng
        _SAFETY_MODULE.clear_memory()
        _FRAME_STACK.clear()
        x, y, theta = 0.0, 0.0, 0.0
        _UNWEDGE_HIDDEN = None
        _UNWEDGE_PREV_ACTION = None
    
    is_stuck = (obs[17] == 1)
    if is_stuck:
        _SAFETY_MODULE.flag_stuck(x, y)
        if _UNWEDGE_HIDDEN is None:
            _UNWEDGE_HIDDEN = torch.zeros(1, 1, _UNWEDGER.hidden_size)
            _UNWEDGE_PREV_ACTION = torch.tensor([0]) 

        with torch.no_grad():
            logits, _, next_hidden = _UNWEDGER(_UNWEDGE_PREV_ACTION, _UNWEDGE_HIDDEN)
            _UNWEDGE_HIDDEN = next_hidden

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
            _UNWEDGE_PREV_ACTION = action 

        _FRAME_STACK.clear()
        update_positional_data(action.item(), action.item() == 2)
        return ACTIONS[action.item()]

    _UNWEDGE_HIDDEN = None
    _UNWEDGE_PREV_ACTION = None

    if len(_FRAME_STACK) == 0:
        for _ in range(4): _FRAME_STACK.append(obs.copy())
    else:
        _FRAME_STACK.append(obs.copy())

    stacked_obs = np.concatenate(_FRAME_STACK, axis=0)
    state_tensor = torch.from_numpy(stacked_obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        raw_logits = _SAC_ACTOR(state_tensor)
        safe_logits = _SAFETY_MODULE.mask_logits(raw_logits, x, y, theta)
        
        probs = F.softmax(safe_logits, dim=-1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()
    
    update_positional_data(action.item(), False)
    return ACTIONS[action.item()]