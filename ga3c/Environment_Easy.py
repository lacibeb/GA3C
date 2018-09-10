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
from EnvironmentGYM import Environment as Env
from Super_Easy_Game import Super_Easy_Game

class Environment(Env):
    def __init__(self):
        self.game = Super_Easy_Game(Config.GAME, Config.CONTINUOUS_INPUT)
        # TODO: only try
        # https://github.com/openai/gym/issues/494
        # conda install - c anaconda pyopengl
        # conda install -c conda-forge xvfbwrapper
        # force true clears directory
        # export DISPLAY=:0.0 in etc/environment
        #self.game = gym.wrappers.Monitor(self.game, 'pics/', force=True, mode='rgb_array', video_callable=lambda episode_id: True)

        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.reset()

        if Config.CONTINUOUS_INPUT:
            self.action_dim = self.game.action_dim
            self.action_bound = self.game.action_bound
        else:
            self.action_dim = self.game.action_dim

        self.state_dim = self.game.state_dim

