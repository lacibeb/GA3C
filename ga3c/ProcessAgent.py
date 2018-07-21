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

# pyperconfig do not need game manager
if Config.GAME == 'pyperrace':
    from PyperEnvironment import Environment
else:
    from Environment import Environment

from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()

        # change player
        self.env.player = 'agent'

        # countinous or discrate input selection
        if Config.CONTINUOUS_INPUT:
            self.num_actions = self.env.get_num_actions()
        else:
            self.num_actions = Config.CONTINUOUS_INPUT_PARTITIONS

        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)
        self.time_count = 0


    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            if Config.REWARD_CLIPPING:
                r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            # without intermediate rewards
            if Config.DISCOUNTING:
                reward_sum = discount_factor * reward_sum
                # with intermediate rewards
                if Config.USE_INTERMEDIATE_REWARD:
                    reward_sum = discount_factor * reward_sum + r
                else:
                    experiences[t].reward = reward_sum
        # return experiences[:-1]
        return experiences

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        x2_ = np.array([exp.next_state for exp in experiences])
        done_ = np.array([exp.done for exp in experiences])
        if Config.CONTINUOUS_INPUT:
            # continuous action
            a_ = np.array([exp.action for exp in experiences])
            a_ = np.reshape(a_, newshape=[len(a_), 1])
        else:
            #discreate action
            # for exp in experiences:
                #  print ("e " + str(exp.action))
            a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_, x2_, done_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            # print( "self.actions " + str(prediction) + " " + str(np.sum(prediction)))
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        self.time_count = 0
        reward_sum = 0.0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0)  # 0 == NOOP
                continue

            prediction, value = self.predict(self.env.current_state)
            if Config.CONTINUOUS_INPUT:
                action = prediction[0]
                env_action = action
            else:
                action = self.select_action(prediction)
                # converting discrate action to continuous
                # converting -1 .. 1 to fixed angles
                env_action = self.convert_action_discrate_to_angle(action)

            reward, done = self.env.step(env_action)

            reward_sum += reward
            exp = Experience(self.env.previous_state, action, prediction, reward, self.env.current_state, done)
            experiences.append(exp)

            if done or self.time_count == Config.TIME_MAX:
                #terminal_reward = 0 if done else value
                # with pyperrace the final reward is always in last step, it always plays until the end
                terminal_reward = reward
                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_, x2_, done_ = self.convert_data(updated_exps)

                print(str(a_.transpose()))
                print(str(r_))
                yield x_, r_, a_, x2_, done_, reward_sum

                # reset the tmax count
                self.time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            self.time_count += 1

    def run(self):
        # print("process started: " + str(self.id))
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))
        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, x2_, done_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_, x2_, done_))
                # print("shape_x " + str(x_.shape[0]))
                # print("qsize: " + str(self.training_q.qsize()))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))

    @staticmethod
    def convert_action_angle_to_discrate(action):
        # convert action continous angle to prediction
        # two nearest action probability will be bigger, others will be 0
        # from the two nearest, probabilities are linear
        prediction = [0.0]*Config.CONTINUOUS_INPUT_PARTITIONS
        for i in range(Config.CONTINUOUS_INPUT_PARTITIONS):
            error = i - (action + 1) / (2 / Config.CONTINUOUS_INPUT_PARTITIONS)
            if abs(error) > 1.0:
                prediction[i] = 0
            else:
                if error < 0.0:
                    if i == Config.CONTINUOUS_INPUT_PARTITIONS - 1:
                        prediction[i] = 1.0 + error
                        prediction[0] = -1.0 * error
                    else:
                        prediction[i] = 1.0 + error
                        prediction[i + 1] = -1.0*error
        # print(str(action) + " " + str(discrate_action) + " " + str(prediction))
        return prediction

    @staticmethod
    def convert_action_discrate_to_angle(action):
        action = (2 / Config.CONTINUOUS_INPUT_PARTITIONS) * action - 1
        return action

