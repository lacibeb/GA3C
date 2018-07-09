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

    def run_episode(self):
        self.env.reset()

        # human reference
        self.env.steps_with_reference()

        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0


        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0)  # 0 == NOOP
                continue

            # prediction, value = self.predict(self.env.current_state)
            # arcade
            # action = self.select_action(prediction)
            # contonuous

            # human reference action
            env_action, player = self.env.get_ref_step(time_count, Config.TIME_MAX)
            # action in -1 .. 1
            # add randomness to it
            env_action = action + np.random.uniform(0.03, -0.03)

            reward, done = self.env.step(env_action)

            if Config.CONTINUOUS_INPUT:
                pass
            else:
                action, prediction = self.convert_action_angle_to_discrate(env_action)

            reward_sum += reward
            exp = Experience(self.env.previous_state, action, prediction, reward, done)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                #terminal_reward = 0 if done else value
                # with pyperrace the final reward is always in last step, it always plays until the end
                terminal_reward = reward
                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1
