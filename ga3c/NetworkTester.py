
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
        count = 0
        while self.exit_flag.value == 0:
            count += 1
            x_ = []
            y_ = []
            ca_ = []
            cv_ = []
            # forward velocity
            for v in [[50, 0], [50, 50], [0, 50], [-50, 50], [-50, 0], [-50, -50], [0, -50], [50, -50]]:

                for i in range(0, 1800, 60):
                    for j in range(0, 1500, 60):
                        current_state = [v[0]/400.0, v[1]/400.0, i/900.0-1, j/900.0-1]
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
                        x_.append(i)
                        y_.append(j)
                        ca_.append(env_action)
                        cv_.append(value)
                if use_matplotlib:
                    # x_ = np.array(x_); y_ = np.array(y_); c_ = np.array(c_);
                    # plt.plot([x for x in x_], [y for y in y_], [c for c in c_])
                    plt.imshow(self.trk_pic)
                    plt.scatter(x_,y_, 0.3, c=ca_)
                    plt.pause(0.001)
                    plt.draw()
                    plt.savefig('./pics/st_' + str(count) + '_action_' + str(v[0]) + "_" + str(v[1]) + '.tif')
                    plt.clf()
                    plt.imshow(self.trk_pic)
                    plt.scatter(x_,y_, 0.3, c=cv_)
                    plt.pause(0.001)
                    plt.draw()
                    plt.savefig('./pics/st_' + str(count) + '_value_' + str(v[0]) + "_" + str(v[1]) + '.tif')
                plt.clf()

