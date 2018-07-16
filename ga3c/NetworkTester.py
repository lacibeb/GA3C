
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
        self.prediction_q = prediction_q
        self.trk_pic = mpimg.imread('h1.bmp')  # beolvassa a pályát
        self.wait_q = Queue(maxsize=1)


    def run(self):
        print("running")
        while self.exit_flag.value == 0:
            x_ = []
            y_ = []
            c_ = []
            plt.imshow(self.trk_pic)
            # forward velocity
            v = [50,0]
            print("before sleep")
            time.sleep(1.0)
            print("after sleep")
            for i in range(0, 500, 10):
                for j in range(0, 500, 10):
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
                    # color = np.array([int(round((env_action + 1) * 127)), 0, 0], dtype='uint8')
                    color = int(round((env_action + 1) * 127))
                    x_.append(i)
                    y_.append(1500-j)
                    c_.append(color)
            if use_matplotlib:
                # x_ = np.array(x_); y_ = np.array(y_); c_ = np.array(c_);
                # plt.plot([x for x in x_], [y for y in y_], [c for c in c_])
                plt.scatter(x_,y_, c=c_)
                plt.pause(0.001)
                plt.draw()
                plt.savefig('try1.png')
                print('saved try')
