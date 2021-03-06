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

from multiprocessing import Queue

import time

from Config import Config

# for pyperrace the statedim is comming from game
if Config.GAME == 'pyperrace':
    from PyperEnvironment import Environment
elif Config.GAME == 'Pendulum-v0':
    Config.CONTINUOUS_INPUT = True
    Config.DISCRATE_INPUT = False
    from EnvironmentPend import Environment
elif Config.GAME == 'CartPole-v0':
    Config.CONTINUOUS_INPUT = False
    Config.DISCRATE_INPUT = True
    from EnvironmentPend import Environment
else:
    pass
    # from Environment import Environment

if Config.CONTINUOUS_INPUT:
    if Config.USE_DDPG:
        from NetworkDDPG import Network
    else:
        from NetworkVP import Network
else:
    from NetworkVP_discrate import Network

from ProcessAgent import ProcessAgent
from ProcessHRAgent import ProcessHRAgent
from ProcessStats import ProcessStats
from ThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer

if Config.USE_REPLAY_MEMORY:
    from ThreadReplay import ThreadReplay

if Config.USE_NETWORK_TESTER:
    from NetworkTester import NetworkTester

class Server:
    def __init__(self):
        self.stats = ProcessStats()

        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.replay_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        self.model = Network(Config.DEVICE, Config.NETWORK_NAME,
                                   self.get_num_action(), self.get_state_dim())

        if Config.LOAD_CHECKPOINT:
            try:
                self.stats.episode_count.value = self.model.load()
            except:
                pass
        self.training_step = 0
        self.frame_counter = 0

        self.agents = []
        self.agent_id = 0

        self.predictors = []
        self.trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

        # Initialize replay memory
        if Config.USE_REPLAY_MEMORY:
            self.dynamic_replay_filler = ThreadReplay(self)

        if Config.USE_NETWORK_TESTER:
            self.tester_prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
            self.tester_predictor = ThreadPredictor(self, 0, self.get_state_dim(), self.tester_prediction_q)
            self.network_tester_process = NetworkTester(100, self.tester_prediction_q)

        print("Server initialized")

    def add_agent(self):
        self.agents.append(
            ProcessAgent(self.agent_id, self.prediction_q, self.training_q, self.stats.episode_log_q))
        self.agents[-1].start()
        self.agent_id += 1

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join()
        self.agents.pop()

    def add_hr_agent(self):
        self.agents.append(
            ProcessHRAgent(self.agent_id, self.prediction_q, self.training_q, self.stats.episode_log_q))
        self.agents[-1].start()
        self.agent_id += 1

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors), self.get_state_dim(), self.prediction_q))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    def train_model(self, x_, r_, a_, x2, done, trainer_id):
        self.model.train(x_, r_, a_, x2, done, trainer_id)
        self.training_step += 1
        self.frame_counter += x_.shape[0]

        self.stats.training_count.value += 1
        self.dynamic_adjustment.temporal_training_count += 1

        if Config.TENSORBOARD and self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
            self.model.log(x_, r_, a_, self.training_step)

    def save_model(self):
        self.model.save(self.stats.episode_count.value)

    def main(self):
        self.stats.start()
        self.dynamic_adjustment.start()
        if Config.USE_REPLAY_MEMORY:
            self.dynamic_replay_filler.start()
        if Config.USE_NETWORK_TESTER:
            self.tester_predictor.start()
            self.network_tester_process.start()

        if Config.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        learning_rate_multiplier = (
                                       Config.LEARNING_RATE_END - Config.LEARNING_RATE_START) / Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < Config.EPISODES:
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

            time.sleep(0.01)

        self.stats.exit_flag = True
        self.dynamic_adjustment.exit_flag = True

        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()

        if Config.USE_REPLAY_MEMORY:
            self.dynamic_replay_filler.exit_flag = True
        if Config.USE_NETWORK_TESTER:
            self.network_tester_process.exit_flag = True
            self.tester_predictor.exit_flag = True

    @staticmethod
    def get_state_dim():
        return Environment().get_state_dim()

    @staticmethod
    def get_num_action():
        return Environment().get_num_actions()




