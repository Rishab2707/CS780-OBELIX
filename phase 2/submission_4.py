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
reflex_queue = [] 
push_counter = 0
is_attached = False
last_action = -1  # Keep track of what we just did


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "SAC_actor_params.pth")


    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(18, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 5)
            )

        def forward(self, x):
            logits = self.net(x)
            return logits

    model = Net()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    global reflex_queue, push_counter, is_attached, last_action
        
    _load_once()

    # --- 1. THE PERMANENT BULLDOZER OVERRIDE ---
    # If we already locked onto the box, ignore the neural network entirely.
    if is_attached:
        last_action = 2
        return ACTIONS[2]  # FWD forever until the boundary ends the episode!

    # --- 2. THE ATTACHMENT DETECTION LOGIC ---
    # obs[16] is Infrared, obs[17] is Stuck bit
    if obs[16] == 1.0 and obs[17] == 0.0 and last_action == 2:
        push_counter += 1
        if push_counter >= 2:
            is_attached = True
            last_action = 2
            return ACTIONS[2]
    else:
        # If the IR turns off, or we get stuck (it was a wall!), reset the counter
        push_counter = 0

    # --- 3. THE EMERGENCY UNWEDGE REFLEX ---
    # We hit a wall AND we aren't already escaping
    if obs[17] == 1.0 and len(reflex_queue) == 0:
        dirn = int(rng.choice([0, 4])) # 0 for L45, 4 for R45
        reflex_queue = [dirn, dirn, 2] # Turn, Turn, Forward

    # --- 4. ESCAPE IN PROGRESS ---
    if len(reflex_queue) > 0:
        action_idx = reflex_queue.pop(0)
        last_action = action_idx
        return ACTIONS[action_idx]
    
    import torch
    # --- 5. PHASE 1: NEURAL NETWORK (THE HUNTER) ---
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    action_idx = int(np.argmax(logits))
    last_action = action_idx
    
    return ACTIONS[action_idx]
