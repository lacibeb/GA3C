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

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config
from NetworkVP import Network as NetworkVP

class Network(NetworkVP):
    def __init__(self, device, model_name, num_actions, state_dim):
        super(Network, self).__init__(device, model_name, num_actions, state_dim)

    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.state_dim], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # As implemented in A3C paper

        self.p_d1 = self.dense_layer(self.x, 2048, 'dense11_p')
        self.p_d2 = self.dense_layer(self.p_d1, 4048, 'dense12_p')
        self.p_d3 = self.dense_layer(self.p_d2, 4048, 'dense13_p')
        self.p_d4 = self.dense_layer(self.p_d3, 4048, 'dense14_p')
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])

        self.d1 = self.dense_layer(self.p_d4, 512, 'dense1')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)
        if Config.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                            tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (
                        1.0 + Config.MIN_POLICY * self.num_actions)
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

            self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                            * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                            tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                          self.softmax_p, axis=1)

        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)

        if Config.DUAL_RMSPROP:
            self.opt_p = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

            self.opt_v = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.cost_all = self.cost_p + self.cost_v
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v)
                self.opt_grad_v_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM), v)
                                           for g, v in self.opt_grad_v if not g is None]
                self.train_op_v = self.opt_v.apply_gradients(self.opt_grad_v_clipped)

                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p)
                self.opt_grad_p_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM), v)
                                           for g, v in self.opt_grad_p if not g is None]
                self.train_op_p = self.opt_p.apply_gradients(self.opt_grad_p_clipped)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.opt_grad = self.opt.compute_gradients(self.cost_all)
                self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM), v) for g, v in
                                         self.opt_grad]
                self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_v = self.opt_p.minimize(self.cost_v, global_step=self.global_step)
                self.train_op_p = self.opt_v.minimize(self.cost_p, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.p_d1))
        summaries.append(tf.summary.histogram("activation_n2", self.p_d2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        # summaries = self.misc_tensor_board(summaries)

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

