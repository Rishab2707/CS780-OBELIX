"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # stores the loaded model

def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "rsac_actor.pth")


    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self, state_dim=18, action_dim=5):
            super().__init__()

            self.feat_ext = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.01)
            )
            
            self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
            self.fc_out = nn.Sequential(
                nn.Linear(80, 64),
                nn.LeakyReLU(0.01),
                nn.Linear(64, action_dim)
            )

        def forward(self, state_sequence, hidden_state=None):
            x = self.feat_ext(state_sequence)
            gru_out, new_hidden = self.gru(x, hidden_state)
            
            raw_sonars = state_sequence[:, :, 0:16] 
            combined_features = torch.cat([gru_out, raw_sonars], dim=-1)
            logits = self.fc_out(combined_features)

            probs = F.softmax(logits, dim=-1)

            return probs, None, new_hidden

    model = Net()
    model.load_state_dict(torch.load(wpath, map_location="cpu")['actor_state_dict'])
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
        
    _load_once()
    
    import torch
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    hidden_state = None
    with torch.no_grad():
        probs, _, hidden_state = _MODEL(x, hidden_state)
        action = torch.argmax(probs.squeeze(), dim=-1)
    
    return ACTIONS[action.item()]
