import pyper_env

from Paper_Config import Config as GameConfig

import numpy as np

class Environment:
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
        return GameConfig.STATE_DIM

    def reset(self):
        self.game.reset(GameConfig.SHOW_WINDOW)
        pos, v = self.game.start_game()
        self.current_state = np.array([v[0]/400.0, v[1]/400.0, (pos[0]/900.0)-1, (pos[1]/900.0-1)])
        # scaling state to be between -1 ... 1


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

    def step(self, action):
        # action randomisation
        action = action + np.random.uniform(0.03, -0.03)

        self.check_bounds(action, 1.0, -1.0, True)
        # Game requires input -180..180 int
        action = int(round(action * 180.0))

        # game step
        v_new, pos_new, step_reward, pos_reward = self.game.step(action, GameConfig.SHOW_WINDOW, draw_text='little_reward',
                                                                 player=self.player)

        end, time, last_t_diff, game_reward, game_ref_reward = self.game.getstate()

        # no image, only pos and speed is the observation
        observation = [v_new[0], v_new[1], pos_new[0], pos_new[1]]

        self.previous_state = self.current_state

        self.current_state = np.array([v_new[0]/400.0, v_new[1]/400.0, (pos_new[0]/900.0)-1, (pos_new[1]/900.0-1)])
        # scaling state to be between -1 ... 1


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

