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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config
from Environment import Environment
from Experience import Experience
from ProcessAgent import ProcessAgent

class ProcessHRAgent(ProcessAgent):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessHRAgent, self).__init__(id, prediction_q, training_q, episode_log_q)

        # change player
        self.env.player = 'href'

    def predict(self, state):

        # human reference action
        env_action, player = self.env.get_ref_step(self.time_count, Config.TIME_MAX)
        # action in -1 .. 1
        # add randomness to it
        env_action = env_action + np.random.uniform(0.03, -0.03)
        env_action = self.env.check_bounds(env_action, 1.0, -1.0, True)
        # print("env_a: " + str(env_action))
        prediction = self.convert_action_angle_to_discrate(env_action)

        value = None
        return prediction, value

    def run_episode(self):
        # human reference
        self.env.steps_with_reference()

        super(ProcessHRAgent, self).run_episode()

