import numpy as np

class Super_Easy_Game():
    def __init__(self, GAME, CONTINUOUS_INPUT):
        self.game = GAME
        self.output = CONTINUOUS_INPUT

        self.action_dim = 1
        self.action_bound = 1.0
        self.state_dim = 1

        self.info = ''
        self.current_state = None
        self.steps = 0
        self.inner_state = np.zeros((1,))
        self.step_reward = 0
        self.reward = 0.0
        self.done = False
        self.start = np.zeros((1,))
        self.reset()

    def reset(self):
        self.step_reward = 0.0
        self.reward = 0.0
        self.done = False
        self.info = 'resetted'
        self.inner_state[0] = np.random.random_sample()*self.action_bound*2 - self.action_bound
        self.start = self.inner_state.copy()
        self.steps = 0

    def step(self, action):
        if self.steps > 200:
            self.done = True
            print('start: ' + str(self.start) + ' end: ' + str(self.inner_state))

        if action is None:
            action = 0.0

        if not self.done:
            if self.game == 'Super_Easy_linear':
                # we can change inner state witch action, basically it is an
                self.inner_state[0] += action*0.01
                self.inner_state[0] = np.clip(self.inner_state[0], -self.action_bound, self.action_bound)

                # absolute function, to converge into zero
                self.step_reward = 1 - np.abs(self.inner_state[0])
                self.reward += self.step_reward
                self.steps += 1

        return self.inner_state, self.step_reward, self.done, self.info

