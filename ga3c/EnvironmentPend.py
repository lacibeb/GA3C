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

import sys

if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc

from Config import Config
from PyperEnvironment import Environment as Env
import gym
from gym import wrappers


class Environment(Env):
    def __init__(self):
        self.game = gym.make(Config.GAME)
        self.game.seed(Config.RANDOM_SEED)

        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.reset()

        if Config.CONTINUOUS_INPUT:
            self.action_bound = self.game.action_space.high

    def get_num_actions(self):
        return self.game.action_space.shape[0]

    def get_state_dim(self):
        return self.game.observation_space.shape[0]

    def reset(self):
        self.game.reset()
        # self.current_state, r, done, info = self.game.step(np.float32([0.0]))
        # self.current_state = np.reshape(self.current_state, -1)

    def step(self, action):
        # action randomisation
        # action = action + np.random.uniform(0.03, -0.03)

        if Config.CONTINUOUS_INPUT:
            self.check_bounds(action, 1.0, -1.0, True)
            # Game requires input -180..180 int
            print('action bef: ' + str(self.action_bound))
            action = action * self.action_bound
            print('action aft: ' + str(action))
            action = [action]

        self.previous_state = self.current_state
        self.current_state, reward, done, info = self.game.step(action)
        self.current_state = np.reshape(self.current_state, -1)
        reward = reward

        return reward, done

