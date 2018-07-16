
from datetime import datetime
from multiprocessing import Process, Queue, Value

# create and display image
use_matplotlib = True


if use_matplotlib:
    #cannot use interactive backend
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

from ProcessAgent import ProcessAgent
import numpy as np
import time

from Config import Config
from Experience import Experience

class NetworkTester(ProcessAgent):
    def __init__(self, id, prediction_q):
        super(ProcessAgent, self).__init__()
        self.exit_flag = Value('i', 0)
        self.id = id

        self.trk_pic = mpimg.imread('h1.bmp')  # beolvassa a pályát

    def run(self):
        while not self.exit_flag:
            x_ = []
            y_ = []
            c_ = []

            # forward velocity
            v = [50,0]
            time.sleep(10.0)
            for i in range(500, step=10):
                for j in range(500, step=10):
                    current_state = [v[0], v[1], i, j]
                    prediction, value = self.predict(current_state)
                    if Config.CONTINUOUS_INPUT:
                        action = prediction[0]
                        env_action = action
                    else:
                        action = self.select_action(prediction)
                        # converting discrate action to continuous
                        # converting -1 .. 1 to fixed angles
                        env_action = self.convert_action_discrate_to_angle(action)
                    color = int(round((env_action + 1) * 127))
                    x_.append(i)
                    y_.append(j)
                    c_.append(color)
            if use_matplotlib:
                plt.plot([x for x in x_], [y for y in y_], [[c for c in c_], 0, 0])
                plt.pause(0.001)
                plt.draw()
                plt.savefig('try1.png')
