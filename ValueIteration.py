from GridWorld import GridWorld
import numpy as np

gamma = 0.05
n = 1
def go():
    # iterate over all states
    global n
    for x in range(W):
        for y in range(H):
            if grid[y, x] == 1:
                continue
            # if abs(env.goal.y - y) > 1 or abs(env.goal.x - x) > 1:
            #     continue
            if y != 4 or x != 0:
                continue
            env.set_pos(x, y)

            # Try all actions and update using max reward
            rs = []
            ss = []
            for a in range(env.action_space.n):
                s, r, _ = env.step(a)
                print(r)
                rs.append(r)
                ss.append(s)
            print()
            a = np.argmax(rs)  
            r = rs[a]          # Highest achieved reward
            s = ss[a]          # Corresponding state
            V_new = r + gamma * V[s]  # Bellmans expectation
            V[y, x] += (V_new - V[y, x])/n  # Running average

    n += 1
    env.display_values(V)

env = GridWorld(wall_pct=0.5, seed=42,render=True, space_fun=go)
grid = env.reset()
H, W = grid.shape
V = np.zeros((H, W))

while True:
    obs = env.process_input()
    if type(obs) == tuple:  # step return
        grid, r, done = obs
    elif type(obs) == np.ndarray:  # reset return
        grid = obs
        V = np.zeros(grid.shape)
        H, W = grid.shape
        V = np.zeros((H, W))
