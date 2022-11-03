from time import sleep
from GridWorld import GridWorld

sts = 0.5  # steps-per-second
env = GridWorld(wall_pct=0.5, render=True)
n = 0
while n < 100:
    # action = env.sample(True)
    action = env.action_space.sample()
    s, r, done = env.step(action)
    if done:
        env.reset()
        # n += 1
        # print(n)
    env.process_input()
    sleep(sts)
    n += 1

print("DONE")