import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from Net import ActorCritic
import gym
from cpprb import ReplayBuffer
from torch.distributions import Categorical
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

class learner(nn.Module):
    def __init__(self, config):
        super(learner, self).__init__()
        self.writer = SummaryWriter(f'runs/learner_logs{time.time()}')
        self.global_step = 0
        self.replay_buffer_size = config.serverconfig.REPLAY_BUFFER_SIZE
        self.test_ep_len = 0 
        
        self.unroll_size = config.baseconfig.UNROLL_SIZE
        self.batch_size = config.serverconfig.BATCH_SIZE
        self.learning_rate = config.serverconfig.LEARNING_RATE  
        self.c_threshold = config.serverconfig.C_THRESHOLD
        self.rho_threshold = config.serverconfig.RHO_THRESHOLD
        self.gamma = config.serverconfig.GAMMA
        self.baseline_loss_scaling = config.serverconfig.BASELINE_LOSS_SCALING
        self.ent_coeff = config.serverconfig.ENTROPY_COEFFICIENT
        
        self.k_epoch = 1  
        
        
        self.env = gym.make('CartPole-v1')
        self.state_shape = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
               
        self.net = ActorCritic(self.state_shape, self.n_action).to("cuda") 
        self.old_net = ActorCritic(self.state_shape, self.n_action).to("cuda")  ### for ppo
        self.old_net.load_state_dict(self.net.state_dict())          
        
        self.memory = ReplayBuffer(self.replay_buffer_size, env_dict = {
                "state": {"shape": (self.unroll_size, self.state_shape)},
                "action": {"shape": (self.unroll_size, 1,)},
                "action_prob": {"shape": (self.unroll_size, 2)}, 
                "reward": {"shape": (self.unroll_size, 1)},
                "done": {"shape": (self.unroll_size, 1)},
                "next_state": {"shape": (self.unroll_size, self.state_shape)},
                "experience_count": {"shape": (1, 1)} 
                # "batch_id": {"shape": (self.unroll_size, 1)},
            })

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
     
    def calculate_vtrace(self, dist, old_dist, dist_actor, old_v, v, next_v, action, reward, done):        
        ratio = torch.exp(dist.log_prob(action) - dist_actor.log_prob(action))
        
        clipped_rho = torch.clamp(ratio, None , max=self.rho_threshold)
        clipped_c = torch.clamp(ratio, None , max=self.c_threshold)

        deltas = reward + ((1.0 - done) * self.gamma * next_v) - v
        deltas = clipped_rho * deltas.squeeze()                
        
        vtrace = torch.zeros_like(deltas)
        advantage = torch.zeros_like(deltas)

        next_vtrace = next_v[-1,:]    
        for t in reversed(range(len(reward))):
            vtrace[t] = v[t] + deltas[t] + self.gamma * clipped_c[t] * (next_vtrace - next_v[t]) * (1 - done[t])
            advantage[t] = clipped_rho[t] * (reward[t] + self.gamma * next_vtrace * (1 - done[t]) - v[t])
            next_vtrace = vtrace[t]

        return vtrace.detach(), advantage.detach()

    def learn(self):
        state = []
        action = []
        action_prob = []
        reward = []
        done = []
        next_state = []
        # try:
        
        # PPO
        self.net.train()
        batch_start_time = time.time()
        try:
            experience = self.memory.sample(self.batch_size)
            batch_time = time.time() - batch_start_time
            self.writer.add_scalar('Timing/batch_time', batch_time, self.global_step)

            for k in range(self.k_epoch): # PPO
                state = torch.tensor(experience['state']).transpose(0,1).to("cuda")
                action = torch.tensor(experience['action']).squeeze().transpose(0,1).to("cuda")
                action_prob = torch.tensor(experience['action_prob']).squeeze().transpose(0,1).to("cuda")
                reward = torch.tensor(experience['reward']).squeeze().transpose(0,1).to("cuda")
                done = torch.tensor(experience['done']).squeeze().transpose(0,1).to("cuda")
                next_state = torch.tensor(experience['next_state']).transpose(0,1).to("cuda")          
                    
                forward_start_time = time.time()
                prob, v = self.net(state)
                _, next_v = self.net(next_state)
                forward_time = time.time() - forward_start_time
                self.writer.add_scalar('Timing/forward_time', forward_time, self.global_step)
                
                v = v.squeeze()
                next_v = next_v.squeeze()
                
                # PPO
                old_prob, old_v = self.old_net(state)
                dist = Categorical(prob)
                dist_old = Categorical(old_prob)
                dist_act = Categorical(action_prob)
                ent = dist.entropy().mean()
            
                v_trace, adv = self.calculate_vtrace(dist, dist_old, dist_act, old_v, v, next_v, action, reward, done)
                v_trace = v_trace.to("cuda")

                critic_loss = self.baseline_loss_scaling * F.mse_loss(v, v_trace)
                entropy_loss = ent * self.ent_coeff
                
                
                ratio = torch.exp(dist.log_prob(action) - dist_act.log_prob(action))
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * adv
                # PPO
                actor_loss = - torch.min(surr1, surr2).mean()
                # actor_loss = -(dist.log_prob(action) * adv).mean()
                # actor_loss = -(dist.log_prob(action) * adv).mean() + entropy_loss
    
                total_loss = actor_loss + critic_loss
                print("total_loss", total_loss)
                backward_start_time = time.time()
                self.optimizer.zero_grad()
                total_loss.backward()
                # actor_loss.backward()
                # critic_loss.backward()
                self.optimizer.step()
                
                backward_time = time.time() - backward_start_time
                self.writer.add_scalar('Timing/backward_time', backward_time, self.global_step)
                self.writer.add_scalar('Loss/total_loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/actor_loss', actor_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/critic_loss', critic_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/critic_loss', critic_loss.item(), self.global_step)
                self.writer.add_scalar('Entropy/entropy', entropy_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/reward', self.test_ep_len, self.global_step)
                self.global_step += 1
        except Exception as e:
            print("not enough experience")
            print(e)
            time.sleep(5)
        self.old_net.load_state_dict(self.net.state_dict())
            
    def test(self):
        state = self.env.reset()[0]
        for i in range(10000):
            self.net.eval()
            prob, v = self.net(torch.Tensor(state).to("cuda"))
            dists = Categorical(prob)
            action = dists.sample().item()
            next_state, reward, done, _, _  = self.env.step(action)
            state = next_state
            self.env.render()
            if done:
                print(i)
                break
        self.test_ep_len = i
        self.env.close()        
        