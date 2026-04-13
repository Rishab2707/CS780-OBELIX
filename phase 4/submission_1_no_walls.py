import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_SAC_ACTOR = None
_UNWEDGER = None
_FRAME_STACK = deque(maxlen=4)
_UNWEDGE_HIDDEN = None
_UNWEDGE_PREV_ACTION = None

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
    global _SAC_ACTOR, _UNWEDGER
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


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _FRAME_STACK, _UNWEDGE_HIDDEN, _UNWEDGE_PREV_ACTION
    _load_once()

    is_stuck = (obs[17] == 1)
    if is_stuck:
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
        return ACTIONS[action.item()]

    _UNWEDGE_HIDDEN = None
    _UNWEDGE_PREV_ACTION = None

    if len(_FRAME_STACK) == 0:
        for _ in range(4): _FRAME_STACK.append(obs.copy())
    else:
        _FRAME_STACK.append(obs.copy())

    stacked_obs = np.concatenate(_FRAME_STACK, axis=0)
    x = torch.from_numpy(stacked_obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _SAC_ACTOR(x)
        
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs.squeeze())
        action = dist.sample()

    return ACTIONS[action.item()]