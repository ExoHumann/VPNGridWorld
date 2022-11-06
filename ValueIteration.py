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
            if grid[y, x] in [1,3]:
                continue  # Skip walls, and goal

            # Try all actions and update using max expected reward
            for a in range(env.action_space.n):
                env.set_pos(x, y)               # Reset position
                s1, r1, _ = env.step(a)         # Take action
                Q[y, x, a] = r1 + gamma * V[s1] # Belmann's expectation equation
            V_new[y, x] = max(Q[y, x, :])       # Max valued across actions
    V = V_new
    env.display_values(V)

env = GridWorld(wall_pct=0.5, seed=42, render=True, non_diag=True, space_fun=go)
# grid = env.reset()
grid = env.reset_to(TUHE)
H, W = grid.shape
# V = np.random.uniform(-10, 10, (H, W))
V = np.zeros(grid.shape)
V = np.ones(grid.shape) * 2
V[tuple(*np.argwhere(TUHE==3))] = 0.0  # Must do this for proper convergence
Q = np.empty((*V.shape, env.action_space.n))  # (action, y, x)

while True:
    obs = env.process_input()
    if type(obs) == tuple:  # step return
        grid, r, done = obs
    elif type(obs) == np.ndarray:  # reset return
        grid = obs
        V = np.random.uniform(-10, 10, (H, W))
        H, W = grid.shape
        V = np.zeros((H, W))
