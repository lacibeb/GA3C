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

        # Initialize target network weights
        self.actor.update_target_network(self.sess)
        print("target actor initialised")
        self.critic.update_target_network(self.sess)
        print("target critic initialised")

    def _create_graph(self):
        # action input for critic output for actor
        self.action_index = tflearn.input_data(shape=[None, self.num_actions], name='critic_action_input')
        # state input
        self.x = tflearn.input_data(shape=[None, self.state_dim], name='critic_input')

        # critic out reference
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.action_bound = 1.0
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        self.actor_lr = Config.actor_lr
        self.critic_lr = Config.critic_lr
        self.tau = Config.tau
        self.gamma = Config.gamma
        self.actor = ActorNetwork(self.state_dim, self.num_actions, self.action_bound,
                                  self.actor_lr, self.tau, self.x)

        print("actor created")

        self.critic = CriticNetwork(self.state_dim, self.num_actions,
                                    self.critic_lr, self.tau, self.gamma,
                                    self.actor.get_num_trainable_vars(), self.x, self.action_index, self.y_r)

        print("critic created")

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.num_actions))

        print("actor noise created")


    def train(self, x, y_r, a, x2, done, trainer_id):
        self.train_DDPG(x, a, y_r, done, x2)

    def train_DDPG(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        # Calculate targets
        target_q = self.critic.predict_target(self.sess, s2_batch, self.actor.predict_target(self.sess, s2_batch))

        batch_size = np.size(t_batch)
        y_i = r_batch

        #terminal is a boolen 1d array
        for k in range(batch_size):
            if t_batch[k]:
                pass
            else:
                y_i[k] = (r_batch[k] + self.critic.gamma * target_q[k][0])

        #y_i = np.reshape(y_i, (batch_size, 1))

        # Update the critic given the targets
        predicted_q_value, _ = self.critic.train(self.sess, s_batch, a_batch, y_i)

        # Update the actor policy using the sampled gradient
        a_outs = self.actor.predict(self.sess, s_batch)

        # gradienseket ezzel kiolvassa a tensorflow graph-ból és visszamásolja
        grads = self.critic.action_gradients(self.sess, s_batch, a_outs)
        # print("grads: " + str(grads))
        self.actor.train(self.sess, s_batch, grads[0])

        # Update target networks
        self.actor.update_target_network(self.sess)
        self.critic.update_target_network(self.sess)
        return np.amax(predicted_q_value*100)

    def predict_p_and_v(self, x):
        # feed_dict={self.x: x}
        action = self.actor.predict(self.sess, x)
        # it seems to be not used, this done to have no dimension error
        value = action
        return action, value

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs = inputs
        self.out, self.scaled_out = self.create_actor_network(scope='actor')

        self.network_params = tf.trainable_variables()

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

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim], name='actor_action_grad')

        # Combine the gradients here
        # TODOdone:  miért minus az action gradient?
        # http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
        self.actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient, name='actor_grads')

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params), name='actor_optimize')

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

        # initialise variables
        # init = tf.global_variables_initializer()
        # self.sess.run(init)

        # writer = tf.summary.FileWriter(args['summary_dir'], self.sess.graph)
        # writer.close()

    def create_actor_network(self, scope='actor'):
        with tf.name_scope(scope):
            net1 = tflearn.fully_connected(self.inputs, 400, name='actor_fc1')
            net2 = tflearn.layers.normalization.batch_normalization(net1, name='actor_norm1')
            net3 = tflearn.activations.relu(net2)
            net4 = tflearn.fully_connected(net3, 100, name='actor_fc2')
            net5 = tflearn.layers.normalization.batch_normalization(net4, name='actor_norm2')
            net6 = tflearn.activations.relu(net5)
            net7 = tflearn.fully_connected(net6, 30, name='actor_fc3')
            net8 = tflearn.layers.normalization.batch_normalization(net7, name='actor_norm3')
            net9 = tflearn.activations.relu(net8)
            net10 = tflearn.fully_connected(net9, 10, name='actor_fc4')
            net11 = tflearn.layers.normalization.batch_normalization(net10, name='actor_norm4')
            net12 = tflearn.activations.relu(net11)
            """ 
            net = tflearn.fully_connected(net, 30)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            net = tflearn.fully_connected(net, 30)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
            """
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(
                net12, self.action_dim, activation='tanh', weights_init=w_init, name='actor_output')
            # Scale output to -action_bound to action_bound

            scaled_out = tf.multiply(out, self.action_bound)
            # scaled_out = np.sign(out)
            return out, scaled_out

    def train(self, sess, inputs, a_gradient):
        sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, sess, inputs):
        prediction = sess.run(self.out, feed_dict={
                self.inputs: inputs})

        if Config.add_uncertainity:
            return prediction + self.uncertanity()
        if Config.add_OUnoise:
            return prediction + self.actior_noise
        return prediction


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

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, inputs, action, y_r):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma

        # Create the critic network

        self.inputs = inputs
        self.action = action
        self.out = self.create_critic_network(scope='critic')

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs = inputs
        self.target_action = action
        self.target_out = self.create_critic_network(scope='critic_target')

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
                learning_rate=self.learning_rate,
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
        with tf.name_scope(scope):
            # inputs from higher level
            # inputs = tflearn.input_data(shape=[None, self.state_dim], name='critic_input')
            net = tflearn.fully_connected(self.inputs, 300, name='critic_fc1')
            net = tflearn.layers.normalization.batch_normalization(net, name='critic_norm1')
            net = tflearn.activations.relu(net)
            t1 = tflearn.fully_connected(net, 200, name='critic_fc2')

            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            # inputs from higher level
            # action = tflearn.input_data(shape=[None, self.action_dim], name='critic_action_input')
            t2 = tflearn.fully_connected(self.action, 200, name='critic_norm2')
            add_t2 = tf.add(tf.matmul(self.action, t2.W), t2.b, name='critic_t2_add')

            net = tflearn.activation(tf.add(tf.matmul(net, t1.W), add_t2), activation='relu', name='critic_relu')

            net = tflearn.fully_connected(net, 90)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            net = tflearn.fully_connected(net, 40)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            net = tflearn.fully_connected(net, 20)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(net, 1, weights_init=w_init, name='critic_output')
            # self.model = model = tflearn.DNN(out)

            return out

    def train(self, sess, inputs, action, predicted_q_value):
        with tf.variable_scope('critic'):
            return sess.run([self.out, self.train_op], feed_dict={
                self.inputs: inputs,
                self.action: action,
                self.cr_learning_rate: self.learning_rate,
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
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
