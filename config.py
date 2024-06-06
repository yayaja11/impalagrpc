class BaseConfig:
    UNROLL_SIZE = 100

class ServerConfig:    
    REPLAY_BUFFER_SIZE = 2000
    LEARNING_RATE = 0.0006    
    BATCH_SIZE = 64
    LEARNING_FREQ = 5
    C_THRESHOLD = 1.0
    RHO_THRESHOLD = 1.0
    GAMMA = 0.98
    BASELINE_LOSS_SCALING = 0.5
    ENTROPY_COEFFICIENT = 0.01
    
class ClientConfig:
    PARAMETER_REQUEST_INTERVAL = 5
    NUM_ACTOR = 4

class Config:
    clientconfig = ClientConfig
    serverconfig = ServerConfig
    baseconfig = BaseConfig