from time import sleep
from GridWorld import GridWorld

sts = 0.01  # steps-per-second
env = GridWorld(wall_pct=0.5)
n = 0
while True:
    # action = GW.sample(True)
    action = env.action_space.sample()
    s, r, done = env.step(action)
    if done:
        env.reset()
        # n += 1
        # print(n)
    env.process_input()
    sleep(sts)