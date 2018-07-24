from EnvironmentPend import Environment as Env
import numpy as np

env = Env()

env.reset()

s2, r,terminal, info = env.game.step(np.float32([10.0]))

print(s2)
print(r)
print(terminal)
print(info)


s2, r = env.step(np.float32([-10.0]))

print(s2)
print(r)

s2, r = env.step(np.float32([10.0]))

print(s2)
print(r)

s2, r = env.step(np.float32([-10.0]))

print(s2)
print(r)
