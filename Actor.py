### <- 지울것
# 남길것

import torch
import torch.nn as nn
from Net import ActorCritic
import gym
from torch.distributions import Categorical
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.env = gym.make('CartPole-v1')
        state_shape = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        
        self.act_count = 0
        self.net = ActorCritic(state_shape, n_action) 
        
    def act(self, state):
        self.act_count += 1
        prob, v = self.net(torch.Tensor(state)) ### torch
        dists = Categorical(prob)        
        action = dists.sample()
        return action, v



def main():
    agent = Agent(100,100)
    for _ in range(4):
        state  = agent.env.reset()[0]
        
        list_state = np.zeros(shape=(UNROLL_SIZE, 4))
        list_next_state = np.zeros(shape=(UNROLL_SIZE, 4))
        list_reward =np.zeros(shape=(UNROLL_SIZE, 1))
        list_done =np.zeros(shape=(UNROLL_SIZE, 1))
        list_truncated = np.zeros(shape=(UNROLL_SIZE, 1))
        list_action =np.zeros(shape=(UNROLL_SIZE, 1))
        list_action_prob = np.zeros(shape=(UNROLL_SIZE,2))
        
        
        for t in range(100):
            print("no")
            action_prob, v = agent.net(torch.Tensor(state))
            dists = Categorical(action_prob)
            action = dists.sample().item()
            next_state, reward, done, truncated, _ = agent.env.step(action)
            
            list_state[t, :] = state
            list_next_state[t,:] = next_state
            list_reward[t,:] = reward
            list_done[t,:] = done
            list_truncated[t,:] = truncated
            list_action[t,:] = action
            list_action_prob[t,:] = action_prob.detach()
            
            state = next_state
            if done:

                break
            
        agent.save_traj(list_state, list_action, list_action_prob, 
                        list_reward, list_done, list_next_state)
        # import pdb; pdb.set_trace()


        list_state = np.zeros(shape=(100, 4))
        list_next_state = np.zeros(shape=(100, 4))
        list_reward =np.zeros(shape=(100, 1))
        list_done =np.zeros(shape=(100, 1))
        list_truncated = np.zeros(shape=(100, 1))
        list_action =np.zeros(shape=(100, 1))
        list_action_prob = np.zeros(shape=(100,2))
                        
            

if __name__ == '__main__':
    main()
