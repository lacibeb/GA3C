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


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        print("thread started: " + str(self.id))
        while not self.exit_flag:
            if Config.USE_REPLAY_MEMORY:
                # move experiences to replay memory
                while self.server.training_q.qsize() > Config.MIN_QUEUE_SIZE:
                    print(str(self.server.training_q.qsize()))
                    x_, r_, a_, x2_, done_ = self.server.training_q.get()
                    # replay memory uses experiences individually
                    for i in range(x_.shape[0]):
                        self.server.replay_buffer.add(x_[i], a_[i], r_[i], done_[i], x2_[i])

                # if enough experience in replay memory than get a random sample
                if self.server.replay_buffer.size() > Config.TRAINING_MIN_BATCH_SIZE:
                    x__, a__, r__, done__, x2__ = \
                        self.server.replay_buffer.sample_batch(Config.TRAINING_MIN_BATCH_SIZE)
                    if Config.TRAIN_MODELS:
                        self.server.train_model(x__, r__, a__, x2__, done__, self.id)
            else:
                batch_size = 0
                while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                    x_, r_, a_, x2_, done_ = self.server.training_q.get()
                    if batch_size == 0:
                        x__ = x_; r__ = r_; a__ = a_; x2__ = x2_; done__ = done_
                    else:
                        x__ = np.concatenate((x__, x_))
                        r__ = np.concatenate((r__, r_))
                        a__ = np.concatenate((a__, a_))
                        x2__ = np.concatenate((x2__, x2_))
                        done__ = np.concatenate((done__, done_))
                    batch_size += x_.shape[0]

                if Config.TRAIN_MODELS:
                    self.server.train_model(x__, r__, a__, x2__, done__, self.id)
