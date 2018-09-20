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

from NetworkVP_original import Network as NetworkVP

# from tensorflow.python import debug as tf_debug

class Network(NetworkVP):
    def __init__(self, device, model_name, num_actions, state_dim):
        self.state_dim = state_dim

        super(Network, self).__init__(device, model_name, num_actions, state_dim)

    def _create_graph(self):
        self._core_graph()
        self._postproc_graph()
        self._opt_graph()

    def _core_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.state_dim], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # As implemented in A3C paper

        self.DNN = self._create_DNN(self.x, Config.DENSE_LAYERS)
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])

        self.logits_v = tf.squeeze(self.dense_layer(self.DNN, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

    # creating dnn from config with fullyconnected layers
    @staticmethod
    def _create_DNN(input, layers, name = ''):
        # creating dense layers as in config
        layercount = 0
        for layer in layers:
            layercount += 1
            if layercount == 1:
                output = Network.dense_layer(input, layer, name + 'dense1_' + str(layercount) + '_p')
            else:
                output = Network.dense_layer(output, layer, name + 'dense1_' + str(layercount) + '_p')
            print(str(layercount) + '. layer: ' + str(layer) + ' dense neurons')
        return output

    def _postproc_graph(self):
        # output, action
        self.logits_p = self._create_angle_output(self.DNN, self.num_actions, 'logits_p', func=tf.nn.sigmoid)

        #output softmax
        self.softmax_p = self.logits_p
        self.log_softmax_p = self.logits_p
        self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

        self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
        self.cost_p_2 = -1 * self.var_beta * \
                    tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)

        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)
        
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

        summaries.append(tf.summary.histogram("activation_lastdense", self.DNN))

        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        # summaries = self.misc_tensor_board(summaries)

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def _create_angle_output(self, input, out_dim, name, func=tf.nn.sigmoid):
        with tf.variable_scope(name):
            # https: // stats.stackexchange.com / questions / 218407 / encoding - angle - data -
            # for -neural - network

            # adding two neuron to output
            x = self.dense_layer(input, out_dim, 'out_x', func)
            y = self.dense_layer(input, out_dim, 'out_y', func)

            # only works for sigmoid
            # TODO other activation
            x = tf.add(x, -0.5)
            y = tf.add(y, -0.5)

            # converting to angle
            output = tf.divide(tf.atan2(y, x), np.pi)

        return output








