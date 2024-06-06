# from concurrent import futures
# import grpc
# import impala_pb2
# import impala_pb2_grpc

# class ImpalaService(impala_pb2_grpc.ImpalaServiceServicer):
#     def __init__(self) -> None:
#         self.data_count: int = 0
#         self.parameters: list[str] = ['a', 'b', 'c', 'd', 'e']
#         self.current_parameter_index: int = 0

#     def CheckConnection(self, request, context) -> impala_pb2.ConnectionStatus:
#         return impala_pb2.ConnectionStatus(connected=True)

#     def SendExperience(self, request, context) -> impala_pb2.Empty:
#         self.data_count += request.count
#         if self.data_count >= 1000:
#             print(f"Total experiences received: {self.data_count}")
#             self.data_count = 0  # Reset after processing
#         return impala_pb2.Empty()

#     def RequestParameter(self, request, context)-> impala_pb2.ParameterResponse:
#         parameter = self.parameters[self.current_parameter_index]
#         self.current_parameter_index = (self.current_parameter_index + 1) % len(self.parameters)
#         return impala_pb2.ParameterResponse(parameter=parameter)

# def serve() -> None:
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     impala_pb2_grpc.add_ImpalaServiceServicer_to_server(ImpalaService(), server)
#     server.add_insecure_port('59.14.117.221:50051')
#     server.start()
#     server.wait_for_termination()

# if __name__ == '__main__':
#     serve()
    
    
from concurrent import futures
import grpc
import impala_pb2
import impala_pb2_grpc
import pickle
import dill
from Actor import Agent
import gym
from Net import ActorCritic
from torch import optim
from Learner import learner
import json
import numpy as np
import time
import threading
from config import Config as config


class ImpalaService(impala_pb2_grpc.ImpalaServiceServicer):
    def __init__(self, learner:learner, config):
        self.total_experiences = 0
        self.critic = learner
        self.state_shape = self.critic.state_shape
        self.n_action = self.critic.n_action
        self.start_time = time.time()
        self.learning_freq = config.serverconfig.LEARNING_FREQ
        self.unroll_size = config.baseconfig.UNROLL_SIZE
        
        
        self.learning_schedule()
        
        self.save_weight(self.critic.net.state_dict())
        
    def learning_schedule(self):
        learning_timer = threading.Timer(self.learning_freq, self.critic_learning)
        learning_timer.start()
        critic_test = threading.Timer(self.learning_freq, self.critic.test)
        critic_test.start()
        
    def critic_learning(self):
        self.critic.learn()
        self.save_weight(self.critic.net.state_dict())
        self.learning_schedule()
        
    
    def CheckConnection(self, request:str, context) -> impala_pb2.ConnectionStatus:
        return impala_pb2.ConnectionStatus(connected=True)    

    def convert_experience_to_cpprb(self, experience):
        # batch_id = time.time() - self.start_time        
        list_state = np.zeros(shape=(self.unroll_size, self.state_shape)) ### 한ep돌고 제대로 다시 0으로 체워지는지 체크
        list_next_state = np.zeros(shape=(self.unroll_size, self.state_shape))
        list_reward =np.zeros(shape=(self.unroll_size, 1))
        list_done =np.zeros(shape=(self.unroll_size, 1))
        list_truncated = np.zeros(shape=(self.unroll_size, 1))
        list_action =np.zeros(shape=(self.unroll_size, 1))
        list_action_prob = np.zeros(shape=(self.unroll_size, self.n_action))
        # list_batch_id = np.zeros(shape=(UNROLL_SIZE, 1))
        exp_count = experience.experience_count + 1
        
        for t in range(exp_count):
            list_state[t, :] = experience.state[t].values
            list_next_state[t,:] = experience.next_state[t].values
            list_reward[t,:] = experience.reward[t].value
            list_done[t,:] = experience.done[t].value
            list_action[t,:] = experience.action[t].value
            list_action_prob[t,:] = experience.action_prob[t].probs
            # list_batch_id[t,:] = batch_id

        self.critic.memory.add(state=list_state, action=list_action, action_prob=list_action_prob, 
                reward=list_reward, done=list_done, next_state=list_next_state, experience_count=exp_count)

    def SendExperience(self, request, context):
        self.convert_experience_to_cpprb(request.experience)
        return impala_pb2.Empty()

    def RequestParameter(self, request, context):
        parameter = self.weight
        context.set_compression(grpc.Compression.NoCompression)
        return impala_pb2.ParameterResponse(parameter=parameter)
    
    def save_weight(self, weight):
        weight = dill.dumps(weight)
        self.weight = weight
    
def run_critic():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),  compression=grpc.Compression.Gzip, 
        options=[
            ('grpc.max_receive_message_length', 10 * 1024 * 1024)  
        ])
    critic = learner(config)
    impala_service = ImpalaService(learner=critic, config=config)
    impala_pb2_grpc.add_ImpalaServiceServicer_to_server(impala_service, server)
    server.add_insecure_port('127.0.0.1:50051')
    server.start() 
    
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()

if __name__ == '__main__':
    run_critic()