
import grpc
import impala_pb2
import impala_pb2_grpc
import random
import multiprocessing
import time
import dill
import msgpack

import pdb,sys,os
import gym
from Actor import Agent

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import threading
from config import Config as config

class client():
    def __init__(self, config):
        self.total_reward = 0
        self.unroll_size = config.baseconfig.UNROLL_SIZE
        self.param_interval = config.clientconfig.PARAMETER_REQUEST_INTERVAL
        

    def create_state(self, values):
        state = impala_pb2.State()
        state.values.extend(values)
        return state

    def create_action(self, value):
        action = impala_pb2.Action()
        action.value = value
        return action

    def create_action_prob(self, probs):
        action_prob = impala_pb2.ActionProb()
        action_prob.probs.extend(probs)
        return action_prob

    def create_reward(self, value):
        reward = impala_pb2.Reward()
        reward.value = value
        return reward

    def create_done(self, value):
        if value == 0:
            value = False
        else:
            value = True
        done = impala_pb2.Done()
        done.value = value
        return done

    # grpc 전송을 위한 ExperienceData 생성
    def create_experience_data(self, data, t):
        experience_data = impala_pb2.ExperienceData()
        for s in data[0]:
            experience_data.state.append(self.create_state(s))
        for a in data[1]:
            experience_data.action.append(self.create_action(a[0]))
        for ap in data[2]:
            experience_data.action_prob.append(self.create_action_prob(ap))
        for r in data[3]:
            experience_data.reward.append(self.create_reward(r[0]))
        for d in data[4]:
            experience_data.done.append(self.create_done(d[0]))
        for ns in data[5]:
            experience_data.next_state.append(self.create_state(ns))
        experience_data.experience_count = t
        return experience_data

    def initialize_actor_buffer(self, state_shape, n_action):
        self.np_state = np.zeros(shape=(self.unroll_size, state_shape)) ### 한ep돌고 제대로 다시 0으로 체워지는지 체크
        self.np_next_state = np.zeros(shape=(self.unroll_size, state_shape))
        self.np_reward =np.zeros(shape=(self.unroll_size, 1))
        self.np_done =np.zeros(shape=(self.unroll_size, 1))
        self.np_action =np.zeros(shape=(self.unroll_size, 1))
        self.np_action_prob = np.zeros(shape=(self.unroll_size,n_action))

    def update_actor_buffer(self, state, action, action_prob, reward, done, next_state, t):
        self.np_state[t, :] = state
        self.np_next_state[t,:] = next_state
        self.np_reward[t,:] = reward
        self.np_done[t,:] = done
        self.np_action[t,:] = action
        self.np_action_prob[t,:] = action_prob
        
    def update_parameters(self, actor_id, stub):
        while True:
            try:
                response = stub.RequestParameter(impala_pb2.Empty())
                new_weights = dill.loads(response.parameter)
                self.actor.net.load_state_dict(new_weights)
                # print("Updated parameters received")
            except grpc.RpcError as e:
                print("Failed to retrieve parameters:", e)
            time.sleep(self.param_interval)

    def run_actor(self, actor_id):
        # time.sleep(int(actor_id))
        self.actor_id = actor_id
        channel = grpc.insecure_channel('127.0.0.1:50052', compression=grpc.Compression.Gzip)
        stub = impala_pb2_grpc.ImpalaServiceStub(channel)
        self.actor = Agent()
        
        update_parameters_thread = threading.Thread(target=self.update_parameters, args=(actor_id, stub))
        update_parameters_thread.start()         
        
        while True:
            total_step = 0
            state_shape = self.actor.env.observation_space.shape[0]
            n_action = self.actor.env.action_space.n   
                        
            t_100 = []
            t_100_box = []
            seed = random.randint(0, 10000)
            state = self.actor.env.reset(seed=seed)[0]
            self.initialize_actor_buffer(state_shape, n_action)   
            for t in range(self.unroll_size):
                action_prob, v = self.actor.net(torch.Tensor(state))
                dists = Categorical(action_prob)
                action = dists.sample().item()
                next_state, reward, done, truncated, _ = self.actor.env.step(action)
                self.total_reward += reward
                # ForkablePdb().set_trace()
        
                self.update_actor_buffer(state, action, action_prob.detach(), reward, done, next_state, t)

                total_step += 1
                
                state = next_state
                
                
                experience_count = t 
                if done:
                    seed = random.randint(0, 10000)
                    self.actor.env.reset(seed=seed)
                    t_100.append(t+1)
                    print(f"{self.actor_id} : total reward : {self.total_reward} ")
                    self.total_reward = 0
                    continue

            experience = [self.np_state, self.np_action, self.np_action_prob, self.np_reward, self.np_done, self.np_next_state]
            experience = self.create_experience_data(experience, experience_count)

            
            stub.SendExperience(impala_pb2.ExperienceRequest(count=total_step, experience = experience ,actor_id=actor_id))       
            total_step = 0

if __name__ == '__main__':
    processes = []
    client_manager = client(config)    
    for i in range(1, config.clientconfig.NUM_ACTOR+1):
        p = multiprocessing.Process(target=client_manager.run_actor, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
            

class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin