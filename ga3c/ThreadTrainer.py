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
        print('Thread Trainer '+ str(id) + ': initialized')

    def run(self):
        # print("thread started: " + str(self.id))
        while not self.exit_flag:
            if Config.USE_REPLAY_MEMORY:
                try:
                    x__, a__, r__, done__, x2__ = self.server.replay_q.get(timeout=20)
                except TimeoutError as err:
                    if self.exit_flag:
                        continue
            else:
                batch_size = 0
                while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                    try:
                        x_, r_, a_, x2_, done_ = self.server.training_q.get(timeout=20)
                    except TimeoutError as err:
                        if self.exit_flag: break
                        continue

                    if batch_size == 0:
                        x__ = x_; r__ = r_; a__ = a_; x2__ = x2_; done__ = done_
                    else:
                        x__ = np.concatenate((x__, x_))
                        r__ = np.concatenate((r__, r_))
                        a__ = np.concatenate((a__, a_))
                        x2__ = np.concatenate((x2__, x2_))
                        done__ = np.concatenate((done__, done_))
                    batch_size += x_.shape[0]

            if Config.TRAIN_MODELS and not self.exit_flag and x__.shape[0] > 0:
                print('x__: ' + str(x__))
                # print('r__: ' + str(r__))
                # print('a__: ' + str(a__))
                print('x2__: ' + str(x2__))
                # print('done__: ' + str(done__))
                self.server.train_model(x__, r__, a__, x2__, done__, self.id)
