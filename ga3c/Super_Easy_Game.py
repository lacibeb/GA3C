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
        self.inner_state[0] = np.random.random_sample()*self.action_bound*2.0 - self.action_bound
        self.start = self.inner_state.copy()
        self.steps = 0

    def step(self, action):

        if action is None:
            action = 0.0

        if not self.done:
            if self.game == 'Super_Easy_linear':
                # we can change inner state witch action, basically it is an
                # print('action: ' + str(action) + 'innerstate: ' + str(self.inner_state[0]))
                self.inner_state[0] = self.inner_state[0] + action*0.05
                # print('innerstate: ' + str(self.inner_state[0]))
                if abs(self.inner_state[0]) > 1:
                    self.done = True
                    self.step_reward = -100
                    self.inner_state = np.clip(self.inner_state, 1, -1)
                else:
                    self.step_reward = 1 - np.abs(self.inner_state[0])
                # print('reward: ' + str(self.step_reward))
                # self.inner_state[0] = np.clip(self.inner_state[0], -self.action_bound, self.action_bound)

                # absolute function, to converge into zero
                self.steps += 1
                if self.steps > 200.0:
                    self.done = True
                    # print('start: ' + str(self.start) + ' end: ' + str(self.inner_state))

                self.reward += self.step_reward

        return self.inner_state, self.step_reward, self.done, self.info

