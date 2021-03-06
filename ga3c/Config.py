# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

class Config:

    #########################################################################
    # Game configuration

    # Name of the game, with version (e.g. PongDeterministic-v0)
    # TODO atary option is not working yet
    # TODO atary games works with image input(missing) and dicrate output(this is implemented)
    # GAME = 'PongDeterministic-v0'
    GAME = 'Pendulum-v0'
    # GAME = 'CartPole-v0'
    # GAME = 'pyperrace'
    # Enable to see the trained agent in action
    PLAY_MODE = False
    # Enable to train
    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = False
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 0 

    #########################################################################
    # Number of agents, predictors, trainers and other system settings
    
    # If the dynamic configuration is on, these are the initial values.
    # Number of Agents
    AGENTS = 32
    # Number of human reference Agents from Agents AGENTS=humref+regular
    HUMAN_REF_AGENTS = 0
    # Number of Predictors
    PREDICTORS = 2
    # Number of Trainers
    TRAINERS = 2

    # Device
    DEVICE = 'gpu:0'

    # Enable the dynamic adjustment (+ waiting time to start it)
    DYNAMIC_SETTINGS = True
    DYNAMIC_SETTINGS_STEP_WAIT = 20
    DYNAMIC_SETTINGS_INITIAL_WAIT = 10

    #########################################################################
    # Algorithm parameters

    # Discount factor
    DISCOUNTING = True
    DISCOUNT = 0.99
    
    # Tmax
    # TIME_MAX = 5
    
    # Reward Clipping
    REWARD_CLIPPING = True
    USE_INTERMEDIATE_REWARD = False
    REWARD_MIN = -1
    REWARD_MAX = 1

    # Max size of the queue
    MAX_QUEUE_SIZE = 100
    PREDICTION_BATCH_SIZE = 128

    # Input of the DNN
    STACKED_FRAMES = 4
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84

    # Total number of episodes and annealing frequency
    #EPISODES = 5000000
    #ANNEALING_EPISODE_COUNT = 500000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.01
    BETA_END = 0.01

    # Learning rate
    #LEARNING_RATE_START = 0.0001
    #LEARNING_RATE_END = 0.00000001

    #Network structure
    DENSE_LAYERS = (10, 10, 10, 10)
    # RMSProp parameters
    # if False than ADAM optimizer only for ddpg
    #RMSPROP = True
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1

    # Dual RMSProp - we found that using a single RMSProp for the two cost function works better and faster
    DUAL_RMSPROP = False
    
    # Gradient clipping
    USE_GRAD_CLIP = False
    GRAD_CLIP_NORM = 40.0 
    # Epsilon (regularize policy lag in GA3C)
    LOG_EPSILON = 1e-6
    # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower
    TRAINING_MIN_BATCH_SIZE = 0
    
    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = True
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 1000
    
    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 1
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 1000

    # Results filename
    RESULTS_FILENAME = 'results.txt'
    # Network checkpoint name
    NETWORK_NAME = 'network'

    #########################################################################
    # More experimental parameters here
    
    # Minimum policy
    MIN_POLICY = 0.0
    # Use log_softmax() instead of log(softmax())
    # not used with continuous
    USE_LOG_SOFTMAX = False

    #########################################################################
    # Network selection
    # define input
    DISCRATE_INPUT = True
    CONTINUOUS_INPUT = False

    CONTINUOUS_INPUT_PARTITIONS = 8

    # use ddpg model it works only with continuous input
    USE_DDPG = False
    if USE_DDPG:
        add_uncertainity = False
        add_OUnoise = True

        USE_REPLAY_MEMORY = True
        REPLAY_BUFFER_SIZE = 1000000
        REPLAY_BUFFER_RANDOM_SEED = 12345
        REPLAY_MIN_QUEUE_SIZE = 2
        DDPG_FUTURE_REWARD_CALC = True
        # with DDPG
        # multiplayer to learning rate
        actor_lr = 0.3
        critic_lr = 2
        tau = 0.001
        gamma = 0.99
        DISCOUNTING = False
    else:
        USE_REPLAY_MEMORY = False

    RANDOM_SEED = 12345
    USE_NETWORK_TESTER = False

    # ------------------------------------
    # recommended game specific settings
    #if GAME == 'Pendulum-v0':
    TIME_MAX = 1000
    LEARNING_RATE_START = 0.0003
    LEARNING_RATE_END = 0.0003
    EPISODES = 40000
    ANNEALING_EPISODE_COUNT = 40000
    actor_lr = 1
    critic_lr = 10
    RMSPROP = True

    DISCRATE_TO_CONTINUOUS_CONVERSION = False

