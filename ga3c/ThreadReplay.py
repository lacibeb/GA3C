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

from threading import Thread

import numpy as np

from Config import Config
from replay_buffer import ReplayBuffer

class ThreadReplay(Thread):
    def __init__(self, server):
        super(ThreadReplay, self).__init__()
        self.setDaemon(True)

        self.server = server
        self.exit_flag = False

        self.replay_buffer = ReplayBuffer(buffer_size=Config.REPLAY_BUFFER_SIZE, \
                                          random_seed=Config.REPLAY_BUFFER_RANDOM_SEED)

    def update_stats(self):
        self.server.stats.replay_memory_size = self.replay_buffer.size()

    def run(self):
        #print("thread started: " + str(self.id))
        while not self.exit_flag:
            # if queue is near empty put a batch there
            if self.server.replay_q.qsize() < Config.REPLAY_MIN_QUEUE_SIZE:
                x__, r__, a__, x2__, done__ = \
                    self.replay_buffer.sample_batch(Config.TRAINING_MIN_BATCH_SIZE)
                self.server.replay_q.put((x__, r__, a__, x2__, done__))
                print("put to replay")
            x_, r_, a_, x2_, done_ = self.server.training_q.get()
            print("get from training")
            # replay memory uses experiences individually
            for i in range(x_.shape[0]):
                self.replay_buffer.add(x_[i], a_[i], r_[i], done_[i], x2_[i])
            self.update_stats()
        # cleaning
        self.replay_buffer.clear()
