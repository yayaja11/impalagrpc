import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_shape , 128),
            nn.ReLU(), 
            nn.Dropout(p=0.6),
            nn.Linear(128 , 64),
            nn.ReLU(), 
            nn.Linear(64 , 32),
            nn.ReLU(), 
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1)
            )
        
        self.critic = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(128 , 64),
            nn.ReLU(), 
            nn.Linear(64 , 32),
            nn.ReLU(), 
            nn.Linear(32, 1)
            )  
                  
    def forward(self, x):
        prob = self.actor(x)
        v = self.critic(x)
        return prob, v