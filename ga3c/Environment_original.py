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
from GameManager import GameManager


class Environment:
    def __init__(self):
        self.game = GameManager(Config.GAME, display=Config.PLAY_MODE)
        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = Queue(maxsize=self.nb_frames)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        self.action_dim = self.get_num_actions()

        self.reset()

    @staticmethod
    def _rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def _preprocess(image):
        image = Environment._rgb2gray(image)
        image = misc.imresize(image, [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH], 'bilinear')
        image = image.astype(np.float32) / 128.0 - 1.0
        return image

    def _get_current_state(self):
        if not self.frame_q.full():
            return None  # frame queue is not full yet.
        x_ = np.array(self.frame_q.queue)
        x_ = np.transpose(x_, [1, 2, 0])  # move channels
        return x_

    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()
        image = Environment._preprocess(frame)
        self.frame_q.put(image)

    def get_num_actions(self):
        return self.game.env.action_space.n

    def get_state_dim(self):
        # not used for atars games
        return [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES]

    def reset(self):
        self.total_reward = 0
        self.frame_q.queue.clear()
        self._update_frame_q(self.game.reset())
        self.previous_state = self.current_state = None

    def step(self, action):
        env_action = None

        if Config.CONTINUOUS_INPUT:
            if action is None:
                action = np.zeros(self.action_dim)
            action = self.check_bounds(action, 1.0, -1.0, True)
            # Game requires input -180..180 int
            # print('action bef: ' + str(self.action_bound))
            env_action = action * self.action_bound
            # print('action aft: ' + str(action))

        if Config.DISCRATE_INPUT:
            env_action_array = np.zeros(self.action_dim, np.dtype(int))
            if action is None:
                action = 0
            env_action_array[action] = 1

            # different from games, not implemented correctly
            # only one action (atary)
            env_action = action
            # array of actions (gym)
            # env_action = env_action_array

        print('env action: ' + str(env_action))
        observation, reward, done, _ = self.game.step(env_action)
        print('observation: ' + str(observation))

        self.total_reward += reward
        self._update_frame_q(observation)

        self.previous_state = self.current_state
        self.current_state = self._get_current_state()
        return reward, done
