"""Baseline agent that is totally random. Used for comparison and trying to break the game."""

from time import sleep
from GridWorld import GridWorld

sts = 0.001  # steps-per-second
env = GridWorld(wall_pct=0.6, map=(5, 15, 5, 10), non_diag=True, resetter=0, space_fun=GridWorld.test)
while True:
    # action = env.sample()  # Only valid action
    action = env.action_space.sample()
    s, r, done = env.step(action)
    if done:
        env.reset()
    env.render()
    env.process_input()
    # sleep(sts)

print("DONE")