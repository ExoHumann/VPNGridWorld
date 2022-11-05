from GridWorld import GridWorld
import numpy as np

TUHE = np.array([[1,0,0,0,0,2,0,1,0,1],
                [1,0,0,0,0,0,1,0,1,0],
                [1,1,0,0,1,0,0,1,1,1],
                [0,0,0,0,1,0,0,1,0,1],
                [1,1,0,1,1,0,1,1,1,0],
                [1,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [1,0,1,0,0,1,0,0,0,1],
                [0,1,1,0,0,0,0,0,1,1],
                [1,0,1,0,1,3,1,1,0,1]])

gamma = 0.95
def go():
    global V
    V_new = V.copy()
    for y in range(H):
        for x in range(W):
            if grid[y, x] in [1, 3]:
                continue  # Skip walls
            # if abs(env.goal.y - y) > 1 or abs(env.goal.x - x) > 1:
            #     continue

            # Try all actions and update using max expected reward
            rs, vs = [], []
            for a in range(env.action_space.n):
                env.set_pos(x, y)  # Reset position
                s, r, _ = env.step(a)
                rs.append(r); vs.append(V[s])
            rs, vs = np.array(rs), np.array(vs)
            a = np.argmax(rs + gamma*vs)         # Isolate best action
            V_new[y, x] = rs[a] + gamma * vs[a]  # Bellmans expectation
    V = V_new
    env.display_values(V)

env = GridWorld(wall_pct=0.5, seed=42,render=True, non_diag=True, space_fun=go)
# grid = env.reset()
grid = env.reset_to(TUHE, (5, 0), (5, 9))
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
