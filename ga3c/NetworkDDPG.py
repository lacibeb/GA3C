# ===========================
#   Actor and Critic DNNs
# ===========================

import tensorflow as tf
import numpy as np
import tflearn

from Config import Config
from NetworkVP import Network as NetworkVP

# interface for GA3C
class Network(NetworkVP):
    def __init__(self, device, model_name, num_actions, state_dim):
        super(Network, self).__init__(device, model_name, num_actions, state_dim)

        self._init_target_networks()
        self.logging = 0.0, 0.0

    def _init_target_networks(self):
        # Initialize target network weights
        self.actor.update_target_network(self.sess)
        print("target actor initialised")
        self.critic.update_target_network(self.sess)
        print("target critic initialised")

    def _opt_graph(self):
        # inside actor and critic
        pass

    def _postproc_graph(self):
        # inside actor and critic
        pass

    def _core_graph(self):
        # action input for critic output for actor
        with tf.variable_scope('inputs'):
            self.action_index = tflearn.input_data(shape=[None, self.num_actions], name='actions')
            # state input
            self.x = tflearn.input_data(shape=[None, self.state_dim], name='state')
            # critic out reference
            self.y_r = tf.placeholder(tf.float32, [None], name='Y_r')

        with tf.variable_scope('Learning_vars'):
            self.action_bound = 1.0
            self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

            self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
            self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
            self.q_max = tf.placeholder(tf.float32, name='Q_max', shape=[])
            self.q_avg = tf.placeholder(tf.float32, name='Q_avg', shape=[])

            self.global_step = tf.Variable(0, trainable=False, name='step')

        self.actor_lr = Config.actor_lr
        self.critic_lr = Config.critic_lr
        self.tau = Config.tau
        self.gamma = Config.gamma
        print('tau: ' + str(self.tau))
        print('ac_lr' + str(self.actor_lr))
        self.actor = ActorNetwork(self.state_dim, self.num_actions, self.action_bound,
                                  self.actor_lr, self.tau, self.x)

        print("actor created")

        self.critic = CriticNetwork(self.state_dim, self.num_actions,
                                    self.critic_lr, self.tau, self.gamma,
                                    self.actor.get_num_trainable_vars(), self.x, self.action_index, self.y_r)



    def train(self, x, y_r, a, x2, done, trainer_id):
         return self.train_DDPG(x, a, y_r, done, x2)

    def train_DDPG(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        # Calculate targets
        target_q = self.critic.predict_target(self.sess, s2_batch, self.actor.predict_target(self.sess, s2_batch))

        batch_size = np.size(t_batch)

        y_i = r_batch

        if Config.DDPG_FUTURE_REWARD_CALC:
            # terminal is a boolen 1d array
            for k in range(batch_size):
                if t_batch[k]:
                    pass
                else:
                    y_i[k] = (r_batch[k] + self.critic.gamma * target_q[k][0])

        # y_i = np.reshape(y_i, (batch_size, 1))

        # Update the critic given the targets
        predicted_q_value, _ = self.critic.train(self.sess, s_batch, a_batch, y_i, self.learning_rate)

        # Update the actor policy using the sampled gradient
        a_outs = self.actor.predict(self.sess, s_batch)

        # gradienseket ezzel kiolvassa a tensorflow graph-ból és visszamásolja
        grads = self.critic.action_gradients(self.sess, s_batch, a_outs)
        # print("grads: " + str(np.transpose(grads[0][:10])))
        # print("a: " + str(np.transpose(a_batch[:10])))
        # print("s: " + str(np.transpose(s_batch[:10])))
        self.actor.train(self.sess, s_batch, grads[0], self.learning_rate)

        # Update target networks
        self.actor.update_target_network(self.sess)
        self.critic.update_target_network(self.sess)

        # not calling log because server calls it
        print('q_max, q_avg: ' + str(np.amax(predicted_q_value)) + ', '+ str(np.average(predicted_q_value)))
        self.logging = np.amax(predicted_q_value), np.average(predicted_q_value)

    def log(self, x, y_r, a, training_step):
        feed_dict = self.__get_base_feed_dict()
        Q_max, Q_avg = self.logging
        feed_dict.update({self.q_max: Q_max, self.q_avg: Q_avg})
        super(Network, self).log(x, y_r, a, training_step, feed_dict)

    def predict_p_and_v(self, x):
        # feed_dict={self.x: x}
        action = self.actor.predict(self.sess, x)
        # it seems to be not used, this done to have no dimension error
        value = action
        return action, value

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = self.critic.create_tensor_board(summaries)
        summaries = self.actor.create_tensor_board(summaries)
        # summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        # summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))

        # summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        # summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        # summaries.append(tf.summary.scalar("Beta", self.var_beta))
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        # summaries.append(tf.summary.histogram("activation_pd1", self.p_d1))
        # summaries.append(tf.summary.histogram("activation_pd2", self.p_d2))
        # summaries.append(tf.summary.histogram("activation_d2", self.d1))
        # summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        # summaries.append(tf.summary.histogram("activation_p", self.softmax_p))
        # summaries = self.misc_tensor_board(summaries)
        summaries.append(tf.summary.scalar("Q_max", self.q_max))
        summaries.append(tf.summary.scalar("Q_avg", self.q_avg))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, inputs):
        with tf.variable_scope('Actor_Agent'):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.action_bound = action_bound
            self.learning_rate = learning_rate
            self.tau = tau

            # Actor Network
            self.inputs = inputs
            with tf.variable_scope('actor'):
                self.out, self.scaled_out = self.create_actor_network(scope='actor')

                self.network_params = tf.trainable_variables()

            with tf.variable_scope('actor_target'):
                # Target Network
                self.target_inputs = inputs
                self.target_out, self.target_out = self.create_actor_network(
                    scope='actor_target')

                self.target_network_params = tf.trainable_variables()[
                                             len(self.network_params):]

                # Op for periodically updating target network with online network
                # weights
                self.update_target_network_params = \
                    [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                          tf.multiply(self.target_network_params[i], 1. - self.tau))
                     for i in range(len(self.target_network_params))]

            with tf.variable_scope('actor_learning'):
                # This gradient will be provided by the critic network
                self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim], name='actor_action_grad')

                # Combine the gradients here
                # TODOdone:  miért minus az action gradient?
                # http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
                self.actor_gradients = tf.gradients(
                    self.out, self.network_params, -self.action_gradient, name='actor_grads')

                self.a_learning_rate = tf.placeholder(tf.float32, name='cr_learning_rate')

                # Optimization Op
                self.optimize = tf.train.AdamOptimizer(self.a_learning_rate). \
                    apply_gradients(zip(self.actor_gradients, self.network_params), name='actor_optimize')

                self.num_trainable_vars = len(
                    self.network_params) + len(self.target_network_params)

                # initialise variables
                # init = tf.global_variables_initializer()
                # self.sess.run(init)

                # writer = tf.summary.FileWriter(args['summary_dir'], self.sess.graph)
                # writer.close()
                if Config.add_OUnoise:
                    self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))

    def create_actor_network(self, scope='actor'):
        self.DNN = super(Network, Network)._create_DNN(self.inputs, Config.DENSE_LAYERS, scope)
        out = super(Network, Network)._create_DNN(self.DNN, (self.action_dim, ), 'actor_output')
        scaled_out = tf.multiply(self.DNN, self.action_bound)
        # scaled_out = np.sign(out)
        return out, scaled_out

    def train(self, sess, inputs, a_gradient, learning_rate):
        sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.a_learning_rate: self.learning_rate*learning_rate
        })

    def predict(self, sess, inputs):
        prediction = sess.run(self.out, feed_dict={
                self.inputs: inputs})
        # print('predddpg: ' + str(prediction))
        if Config.add_uncertainity:
            return prediction + self.uncertanity()
        if Config.add_OUnoise:
            return prediction + self.actor_noise()
        # it is angle, have to be rotated around
        # return np.clip(prediction, self.action_bound, -self.action_bound)
        return self.check_bounds(prediction, self.action_bound, -self.action_bound, True)

    @staticmethod
    def check_bounds(value, posbound, negbound = 0, turnaround = True):
        # if out of bounds then check angle
        if turnaround is False:
            if value < negbound:
                value = negbound
            if value > posbound:
                value = posbound
        else:
            size = posbound - negbound
            if value < negbound:
                value = posbound - ((negbound - value) % size)
            if value > posbound:
                value = ((value - posbound) % size) + negbound
        return value

    def predict_target(self, sess, inputs):
        return sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self, sess):
        sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    # for learning
    def uncertanity(self):
        return np.random.randint(-3, 3, size=1)

    def create_tensor_board(self, summaries):
        #summaries.append(tf.summary.scalar("actor_grads", self.actor_gradients))
        return summaries

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, inputs, action, y_r):
        with tf.variable_scope('Critic_Agent'):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.tau = tau
            self.gamma = gamma

            # Create the critic network

            self.inputs = inputs
            self.action = action
            with tf.variable_scope('critic'):
                self.out = self.create_critic_network()

                self.network_params = tf.trainable_variables()[num_actor_vars:]

            # Target Network
            self.target_inputs = inputs
            self.target_action = action
            with tf.variable_scope('critic_target'):
                self.target_out = self.create_critic_network()

                self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

                # Op for periodically updating target network with online network
                # weights with regularization
                self.update_target_network_params = \
                    [self.target_network_params[i].assign(
                        tf.add(tf.multiply(self.network_params[i], self.tau),
                               tf.multiply(self.target_network_params[i], 1. - self.tau, name='mult_params_' + str(i)),
                               name='add_params_' + str(i)))
                        for i in range(len(self.target_network_params))]

                # Network target (y_i)
                self.predicted_q_value = y_r

            with tf.variable_scope('critic_learning'):
                self.learning_rate = learning_rate
                self.cr_learning_rate = tf.placeholder(tf.float32, name='cr_learning_rate')

                # Define loss and optimization Op
                self.loss = tflearn.mean_square(self.predicted_q_value, self.out)

                # Get the gradient of the net w.r.t. the action.
                # For each action in the minibatch (i.e., for each x in xs),
                # this will sum up the gradients of each critic output in the minibatch
                # w.r.t. that action. Each output is independent of all
                # actions except for one.
                # self.action_grads = tf.gradients(self.out, self.action, name='critic_action_grads')
                # self.action_grads = self.opt_grad_mod
                self.action_grads = tf.gradients(self.out, self.action, name='critic_action_grads')

                if Config.RMSPROP:
                    # no dual optimization
                    # if Config.DUAL_RMSPROP:
                    self.opt_loss = tf.train.RMSPropOptimizer(
                        learning_rate=self.cr_learning_rate,
                        decay=Config.RMSPROP_DECAY,
                        momentum=Config.RMSPROP_MOMENTUM,
                        epsilon=Config.RMSPROP_EPSILON)

                    # gradiens for critic
                    self.opt_grad = self.opt_loss.compute_gradients(self.loss)

                    if Config.USE_GRAD_CLIP:
                        # clipping gradient
                        self.opt_grad_mod = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                                        for g,v in self.opt_grad if not g is None]
                    else:
                        self.opt_grad_mod = self.opt_grad

                    self.train_op = self.opt_loss.apply_gradients(self.opt_grad_mod)
                else:
                    self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.cr_learning_rate).minimize(self.loss)


            # initialise variables
            # init = tf.global_variables_initializer()
            # self.sess.run(init)

            # writer = tf.summary.FileWriter(args['summary_dir'], self.sess.graph)
            # writer.close()

    def create_critic_network(self, scope='critic'):
        # inputs from higher level
        # inputs = tflearn.input_data(shape=[None, self.state_dim], name='critic_input')

        self.state_dnn = super(Network, Network)._create_DNN(self.inputs, Config._CRITIC_STATE_DENSE_LAYERS, scope + '_state')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        # inputs from higher level
        # action = tflearn.input_data(shape=[None, self.action_dim], name='critic_action_input')

        self.action_dnn = super(Network, Network)._create_DNN(self.action, Config._CRITIC_ACTION_DENSE_LAYERS, scope + '_action')

        self.action_and_state = tf.add(self.state_dnn, self.action_dnn, name='critic_added_weights')

        # self.out_dnn = super(Network, Network)._create_DNN(self.action_and_state, Config._CRITIC_OUT_DENSE_LAYERS, scope + '_out')
        self.out_dnn = super(Network, Network)._create_DNN(self.state_dnn, Config._CRITIC_OUT_DENSE_LAYERS,
                                                           scope + '_out_dnn')
        self.out = super(Network, Network)._create_DNN(self.state_dnn, (1, ),
                                                           scope + 'critic_out')
        return self.out

    def train(self, sess, inputs, action, predicted_q_value, learning_rate):
        with tf.variable_scope('critic'):
            return sess.run([self.out, self.train_op], feed_dict={
                self.inputs: inputs,
                self.action: action,
                self.cr_learning_rate: self.learning_rate*learning_rate,
                self.predicted_q_value: predicted_q_value
            })

    def predict(self, sess, inputs, action):
        with tf.variable_scope('critic'):
            return sess.run(self.out, feed_dict={
                self.inputs: inputs,
                self.action: action
            })

    def predict_target(self, sess, inputs, action):
        with tf.variable_scope('critic'):
            return sess.run(self.target_out, feed_dict={
                self.target_inputs: inputs,
                self.target_action: action
            })

    def action_gradients(self, sess, inputs, actions):
        return sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self, sess):
        sess.run(self.update_target_network_params)

    def load(self, sess, path):
        self.saver.load(sess, path)

    def get_variables(self):
        return self.out, self.loss

    def create_tensor_board(self, summaries):
        # summaries.append(tf.summary.scalar("Q_loss", self.loss))
        #for var in tf.trainable_variables():
        #    summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        #self.summary_op = tf.summary.merge(summaries)
        #self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)
        return summaries

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        if self.mu.shape[0] == 1:
            return x[0]
        else:
            return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

