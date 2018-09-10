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
# Import environment
if Config.GAME == 'pyperrace':
    from PyperEnvironment import Environment
elif Config.GAME == 'Pendulum-v0':
    from EnvironmentGYM import Environment
elif Config.GAME == 'CartPole-v0':
    from EnvironmentGYM import Environment
elif Config.GAME == 'Super_Easy_linear':
    from Environment_Easy import Environment
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

        self.num_actions = self.env.get_num_actions()

        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)
        self.time_count = 0
        self.explore_p = Value('i', 0)


    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            if Config.REWARD_RESIZE:
                experiences[t].reward *= Config.REWARD_FACTOR

            if Config.REWARD_CLIPPING:
                r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            else:
                r = experiences[t].reward

            # with intermediate rewards
            if Config.USE_INTERMEDIATE_REWARD:
                experiences[t].reward = r
            else:
                experiences[t].reward = reward_sum + r

            if Config.DISCOUNTING:
                reward_sum = discount_factor * reward_sum

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
        # print(str(state))
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get(timeout=2)

        return p, v

    def select_action(self, actions, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            if Config.EXPLORATION and self.explore_p.value > np.random.rand():
                action = self.env.game.action_space.sample()
            else:
                action = np.random.choice(actions, p=prediction)
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
                self.env.step(None)  # 0 == NOOP
                continue

            print('state: ' + str(self.env.current_state))
            prediction, value = self.predict(self.env.current_state)
            print('pred: ' + str(prediction))

            if Config.DISCRATE_INPUT:
                action = self.select_action(self.actions, prediction)
            else:
                action = prediction

            reward, done = self.env.step(action)
            print('reward: ' + str(reward))
            reward_sum += reward
            exp = Experience(self.env.previous_state, action, prediction, reward, self.env.current_state, done)
            experiences.append(exp)

            if done or self.time_count == Config.TIME_MAX:
                #terminal_reward = 0 if done else value
                # with pyperrace the final reward is always in last step, it always plays until the end
                terminal_reward = reward_sum
                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_, x2_, done_ = self.convert_data(updated_exps)
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
            #try:
            for x_, r_, a_, x2_, done_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_, x2_, done_))
                # print("shape_x " + str(x_.shape[0]))
                # print("qsize: " + str(self.training_q.qsize()))
            #except:
                # if timout occurs it is possible due to end of training
                #continue
            self.episode_log_q.put((datetime.now(), total_reward, total_length))

