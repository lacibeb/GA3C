import pyper_env

from Paper_Config import Config as GameConfig

import numpy as np
from Environment import Environment as Env

class Environment(Env):
    def __init__(self, player='default'):
        self.game = pyper_env.PaperRaceEnv(track_name = GameConfig.track_name, car_name = GameConfig.car_name,\
                                             random_init = GameConfig.random_init, ref_calc = GameConfig.ref_calc, \
                                             save_env_ref_buffer_dir = GameConfig.save_env_ref_buffer_dir, \
                                             save_env_ref_buffer_name = GameConfig.save_env_ref_buffer_name, \
                                             load_env_ref_buffer = GameConfig.load_env_ref_buffer, \
                                             load_all_env_ref_buffer_dir = GameConfig.load_all_env_ref_buffer_dir)

        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        self.game.reset(GameConfig.SHOW_WINDOW)

        self.game.new_player('agent', (1, 0, 0))
        self.game.new_player('href', (0, 1, 0))

        self.player = player

    @staticmethod
    def get_num_actions():
        return GameConfig.ACTION_DIM

    @staticmethod
    def get_state_dim():
        if GameConfig.USE_LIDAR:
            return GameConfig.STATE_DIM + GameConfig.LIDAR_CHANNELS
        else:
            return GameConfig.STATE_DIM

    def reset(self):
        self.game.reset(GameConfig.SHOW_WINDOW)
        pos, v = self.game.start_game()
        self.current_state = np.array([v[0] / 400.0, v[1] / 400.0, (pos[0] / 900.0) - 1, (pos[1] / 900.0 - 1)])

        if GameConfig.USE_LIDAR:
            self.current_state = np.concatenate(self.current_state, self.game.get_lidar_channels())
        # scaling state to be between -1 ... 1


    def step(self, action):
        # action randomisation
        # action = action + np.random.uniform(0.03, -0.03)

        self.check_bounds(action, 1.0, -1.0, True)
        # Game requires input -180..180 int
        action = int(np.round(action * 180.0))

        # game step
        v_new, pos_new, step_reward, pos_reward = self.game.step(action, GameConfig.SHOW_WINDOW, draw_text='little_reward',
                                                                 player=self.player)

        end, time, last_t_diff, game_reward, game_ref_reward = self.game.getstate()

        # if we are using lidar get the channel and also scale to 0..1
        if GameConfig.USE_LIDAR:
            lidar_channels = self.game.get_lidar_channels()/GameConfig.Config.LIDAR_MAX_LENGTH

        # no image, only pos and speed is the observation

        observation = [v_new[0], v_new[1], pos_new[0], pos_new[1]]

        self.previous_state = self.current_state

        self.current_state = np.array([v_new[0]/400.0, v_new[1]/400.0, (pos_new[0]/900.0)-1, (pos_new[1]/900.0-1)])
        # scaling state to be between -1 ... 1

        if GameConfig.USE_LIDAR:
            self.current_state = np.concatenate(self.current_state, self.game.get_lidar_channels())

        done = end

        if GameConfig.reward_based_on_ref:
            reward = last_t_diff*0.01
            if end:
                self.total_reward = game_ref_reward*0.01
                reward += game_ref_reward*0.01
        # position based reward
        else:
            if end:
                self.total_reward = game_reward * 0.01
                reward = game_reward * 0.01
            else:
                reward = step_reward * 0.01
                self.total_reward += reward

        return reward, done

    def steps_with_reference(self):
        self.actions, self.actions_size = self.game.get_steps_with_reference()

    def get_ref_step(self, step, max_steps):
        action, player = self.game.get_ref_step(step, max_steps, self.actions, self.actions_size)
        action = action / 180.0
        return action, player

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