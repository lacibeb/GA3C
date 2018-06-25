import pyper_env
from Paper_Config import Config
import numpy as np

class Environment:
    def __init__(self, player='default'):
        self.game = pyper_env.PaperRaceEnv(track_name = Config.track_name, car_name = Config.car_name,\
                                             random_init = Config.random_init, ref_calc = Config.ref_calc, \
                                             save_env_ref_buffer_dir = Config.save_env_ref_buffer_dir, \
                                             save_env_ref_buffer_name = Config.save_env_ref_buffer_name, \
                                             load_env_ref_buffer = Config.load_env_ref_buffer, \
                                             load_all_env_ref_buffer_dir = Config.load_all_env_ref_buffer_dir)

        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        self.game.reset(Config.SHOW_WINDOW)

        self.game.new_player('agent', (1, 0, 0))
        self.game.new_player('href', (0, 1, 0))

        self.player = player

    @staticmethod
    def get_num_actions():
        return Config.ACTION_DIM

    @staticmethod
    def get_num_states():
        return Config.STATE_DIM

    def reset(self):
        self.game.reset(Config.SHOW_WINDOW)
        pos, v = self.game.start_game()
        self.current_state = [v[0], v[1], pos[0], pos[1]]

    def step(self, action):
        v_new, pos_new, step_reward, pos_reward = self.game.step(action, Config.SHOW_WINDOW, draw_text='little_reward',
                                                                 player=self.player)

        end, time, last_t_diff, game_reward, game_ref_reward = self.game.getstate()

        # no image, only pos and speed is the observation
        observation = [v_new[0], v_new[1], pos_new[0], pos_new[1]]

        self.previous_state = self.current_state

        # TODO it is not markovian because reward depends on past states as well
        self.current_state = np.array([v_new[0], v_new[1], pos_new[0], pos_new[1]])

        done = end

        if Config.reward_based_on_ref:
            if end:
                self.total_reward = game_ref_reward*0.01
                reward = game_ref_reward*0.01
            else:
                reward = last_t_diff*0.01
                self.total_reward += reward
        # position based reward
        else:
            if end:
                self.total_reward = game_reward * 0.01
                reward = game_reward * 0.01
            else:
                reward = step_reward * 0.01
                self.total_reward += reward

        return reward, done
