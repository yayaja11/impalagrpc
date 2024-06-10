class BaseConfig:
    UNROLL_SIZE = 1000

class ServerConfig:    
    REPLAY_BUFFER_SIZE = 1000
    LEARNING_RATE = 0.00006    
    BATCH_SIZE = 128
    LEARNING_FREQ = 1
    C_THRESHOLD = 1.0
    RHO_THRESHOLD = 1.0
    GAMMA = 0.99
    BASELINE_LOSS_SCALING = 0.5
    ENTROPY_COEFFICIENT = 0.01
    
class ClientConfig:
    PARAMETER_REQUEST_INTERVAL = 16
    NUM_ACTOR = 16

class Config:
    clientconfig = ClientConfig
    serverconfig = ServerConfig
    baseconfig = BaseConfig