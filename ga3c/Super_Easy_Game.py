import numpy as np

class Super_Easy_Game():
    def __init__(self, GAME, CONTINUOUS_INPUT):
        self.game = GAME
        self.output = CONTINUOUS_INPUT

        self.action_dim = 1
        self.action_bound = 1
        self.state_dim = 1

        self.info = ''
        self.current_state = None
        self.steps = 0

    def reset(self):
        self.step_reward = 0
        self.reward = 0
        self.done = False
        self.info = 'resetted'
        self.inner_state = np.random.rand(0, self.action_bound*2) - self.action_bound
        self.steps = 0

    def step(self, action):
        if self.steps > 200:
            self.done = True

        if not self.done:
            if self.game == 'Super_Easy_linear':
                # we can change inner state witch action, basically it is an
                self.inner_sate += action*0.01
                self.inner_sate = np.clip(self.inner_sate, -self.action_bound, self.action_bound)

                # absolute function, to converge into zero
                step_reward = 1 - np.abs(self.inner_sate)
                self.reward += step_reward
                self.steps += 1


        return self.inner_state, step_reward, self.done, self.info