""" GridWorld environment for RL
    Run this script for human play

    Common bugs
    - "list index out of range" => random.randint(a, b) is inclusive so replace b with b-1. 
    - Improper grid udating when moving agent about

    TODO
    - ???
"""

from logging import exception
from gym import spaces
import numpy as np
import pygame as pg
import sys, random
from collections import defaultdict

CELL = 48  # Cell width TODO: Makes dependent on map_size but still global...
WHITE, GREY, BLACK = (200, 200, 200), (190, 190, 190), (20, 20, 20)
YELLOW, RED, GREEN = (150, 150, 0), (200, 0, 0), (0, 150, 0)
COLORS = [YELLOW, GREY, GREEN]  # wall, agent, goal

MOVE = [pg.K_d, pg.K_e, pg.K_w, pg.K_q,  # RIGHT, RIGHT_UP, UP, LEFT-UP
        pg.K_a, pg.K_z, pg.K_s, pg.K_c]  # LEFT, LEFT-DOWN, DOWN, RIGHT-DOWN
NUM_ENTITIES = 3
WALL, AGENT, GOAL = range(NUM_ENTITIES)

pg.init()
pg.display.set_caption('GridWorld')
pg.font.init()

class Vec2D():
    """Helper class for flipping coordinates and basic arithmetic.
    pygame draws by v=(x, y) and numpy indexes by p=(y, x)."""

    def __init__(self, x, y=0):
        if type(x) != tuple:
            self.x, self.y = x, y
        else:
            self.x, self.y = x[0], x[1]

    @property
    def v(self):
        return self.x, self.y

    @property
    def p(self):
        return self.y, self.x

    @property
    def sum(self):
        return self.x + self.y

    def __repr__(self):
        return f'Vec2D({self.x}, {self.y})'

    def __add__(self, o):
        return Vec2D(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Vec2D(self.x - o.x, self.y - o.y)

    def __mul__(self, k):
        return Vec2D(k * self.x, k * self.y)

    def __abs__(self):
        return Vec2D(abs(self.x), abs(self.y))

    def __eq__(self, o):
        return self.x == o.x and self.y == o.y

    def __iter__(self):
        for i in [self.x, self.y]:
            yield i

    def __hash__(self):
        return hash((self.x, self.y))


class Display():
    """Handles all rendering"""

    def __init__(self, grid, start, goal, path=None):
        self.grid = grid
        self.H, self.W = grid[WALL].shape
        self.path = path
        self.screen = pg.display.set_mode([self.W * CELL, self.H * CELL])
        self.font = pg.font.Font(None, 25)
        self.reset(start, goal)

    def update(self, pos, before):
        self.grid[AGENT][before.p] = 0
        self.grid[AGENT][pos.p] = 1
        rect = self._draw_rect(before, BLACK)  # Draw prior object (assume never leaving goal)
        pg.display.update(rect)
        if self.path and before in self.path:  # Draw path point
            rect = self._draw_circle(before, CELL >> 3)
            pg.display.update(rect)
        rect = self._draw_circle(pos, CELL >> 2)  # x >> 2 = x / 4     # Draw agent
        pg.display.update(rect)
    


    def reset(self, start, goal): #TODO set agent to start position on given env.
        self.goal = goal
        self.screen.fill(GREY)
        
        for x in range(self.W):
            for y in range(self.H):
                pos = Vec2D(x, y)
                color = YELLOW if self.grid[WALL][pos.p] else BLACK
                self._draw_rect(pos, color)
        
        self._draw_rect(goal, GREEN)          
        if self.path:
            self.draw_path()
        self._draw_circle(start, CELL >> 2)  

        pg.display.flip()

    def draw_policy(self, actions):
        pass

    def draw_path(self):
        for p in self.path:
            self._draw_circle(p, CELL >> 3)  # x >> 3 = x / 8
    
    def _get_color(self, pos):  # Note used currently
        for i in range(NUM_ENTITIES):
            if self.grid[i][pos.p]:
                return COLORS[i]
        return BLACK

    def _draw_rect(self, pos, color):
        pos = pos * CELL + Vec2D(1, 1)
        rect = pg.Rect(pos.v, (CELL - 2, CELL - 2))
        pg.draw.rect(self.screen, color, rect, 0)
        return rect

    def _draw_circle(self, pos, r):
        center = (pos + Vec2D(0.5, 0.5)) * CELL
        return pg.draw.circle(self.screen, GREY, center.v, r)

    # ---- Value iteration ----
    def update_values(self, V):
        """Remove game elements and impose nparray V over grid"""
        assert self.grid[0].shape == V.shape, f"Array dimensions don't match (grid) \
                                            {self.grid.shape} != {V.shape} (V)"
        V = np.round(V, 2)  # Remove decimals up to 2 points
        for x in range(self.W):
            for y in range(self.H):
                if self.grid[WALL][y, x]:
                    continue
                pos = Vec2D(x, y)
                if self.grid[GOAL][y, x]:
                    self._draw_rect(pos, GREEN)
                    continue
                text = self.font.render(str(V[pos.p]), True, WHITE)
                color = [x + V[pos.p]**3 * (y - x) for x, y in zip(BLACK, GREEN)]  # Color gradient
                self._draw_rect(pos, color)
                self._draw_text(text, pos)

        pg.display.flip()

    def _draw_text(self, text, pos):
        center = (pos + Vec2D(0.5, 0.5)) * CELL
        text_rect = text.get_rect(center=center.v)
        self.screen.blit(text, text_rect)


class GridWorld():
    """Gridworld environment following OpenAI functionality

    Helper class Vec2D alleviates syntax strain from numpy indexing by (y, x),
    while pygame instances rects for drawing by (x, y). All rendering is performed
    in helper class Display. 

    Methods:
    reset()         -- Reset all class properties to initial values and generate new grid.
    step(action)    -- Take action in environment and return new grid, reward and done.
    sample()        -- Return a random action. Equivalent to action_space.sample(), but returns non-termination actions.
    close()         -- Quit pygame and if this script is main, terminates script.
    process_input() -- Process input using pygame.

    Important variables:
    grid         -- 3d nparray of hot encoded map with nominal values {0:Wall, 1:Agent, 2:Goal), e.g. grid[WALL][2, 1] = 1 if wall. 
    done         -- bool flag of game termination state.
    action_space -- Discrete of {0:Right, 1:Right-Up, 2:Up, 3:Left-Up, 4:Left, 5:Left-Down, 6:Down, 7:Right-Down}.
    pos          -- Vec2D of current position of agent.
    goal         -- Vec2D of goal position.
    """

    def test(self):
        pass

    def __init__(self, map_size=(5, 10, 5, 10), wall_pct=0.7, rewards=(0.0, 1.0),
                 render=True, seed=None, non_diag=False, space_fun=None):
        """
        Keyword arguments:
        map_size  -- int tuple of (min_x, max_x, min_y, max_y) constraining map size.
        wall_pct  -- float of wall density in grid.
        rewards   -- float tuple (step_penality, win_reward) rewards given after step.
        seed      -- int seed for grid generation and action sampling.
        space_fun -- function to execute. Used for debugging. 
        """
        self.map_size = map_size
        self.wall_pct = wall_pct
        self.render = render
        if seed != None:
            random.seed(seed)
        self.space_fun = space_fun if space_fun else lambda: None

        if not non_diag:
            self.DIRS = [Vec2D(1, 0), Vec2D(1, -1),  # RIGHT, RIGHT-UP
                         Vec2D(0,-1), Vec2D(-1,-1),  # UP, LEFT-UP
                         Vec2D(-1,0), Vec2D(-1, 1),  # LEFT, LEFT-DOWN
                         Vec2D(0, 1), Vec2D(1 , 1)]  # DOWN, RIGHT-DOWN
        else:
            self.DIRS = [Vec2D(1, 0), Vec2D(0, -1),  # RIGHT, UP
                         Vec2D(-1, 0), Vec2D(0, 1)]  # LEFT, DOWN

        self.step_penalty, self.win_reward = rewards

        self.action_space = spaces.Discrete(len(self.DIRS))
        # observation_space defined in reset()

    def reset(self, new_grid=False):
        """Reset class properties (or reset current grid)"""
        self.done = False

        if not new_grid:
            self.grid, start, goal, path = self._generate_grid(self.wall_pct)
            self.observation_space = spaces.Box(0, 1, shape=self.grid.shape, dtype=int)
            self.start = start
            self.pos = start
            self.goal = goal
            
            if self.render:
                self.display = Display(self.grid, self.pos, goal, path)
        else:
            if self.render:
                self.display.update(self.start, self.pos)
            self.pos = self.start

        return self.grid

    def reset_to(self, grid):
        """Set environment to ndarray grid"""
        self.done = False
        self.grid = self._one_hot_encode(grid)
        self.H, self.W = grid.shape
        self.observation_space = spaces.Box(0, 1, shape=self.grid.shape, dtype=int)
        self.pos = Vec2D(tuple(*np.argwhere(self.grid[AGENT].T)))
        self.goal = Vec2D(tuple(*np.argwhere(self.grid[GOAL].T)))

        if self.render:
            self.display = Display(self.grid, self.pos, self.goal)

        return self.grid

    def step(self, action):
        """Take action and return new state (y, x), reward and done 
        Mutate pos, done, grid and display results."""
        # Let terminate = done to reset env after failure
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        # Action from terminal state undefined
        # if self.pos == self.goal:
        #     return [], 0, True

        reward = self.step_penalty
        new_pos = self.pos + self.DIRS[action]

        terminate = self._is_collide(new_pos, self.grid)
        # done = terminate
        if not terminate:
            if self.grid[GOAL][new_pos.p]:
                self.done = True
                reward = self.win_reward
            if self.render:
                self.display.update(new_pos, self.pos)

            # Update grid
            self.grid[AGENT][self.pos.p] = 0
            self.grid[AGENT][new_pos.p] = 1

            self.pos = new_pos

        return self.grid, reward, self.done

    def sample(self):
        """Return random action in action_space"""
        # if legal_only:
        legal_actions = list(filter(lambda dir: not self._is_collide(self.pos + dir, self.grid), self.DIRS))
        action = random.choice(legal_actions)
        return self.DIRS.index(action)
        # return random.randint(0, len(self.DIRS)-1)

    def close(self, total=False):
        """Quit pygame and script if total is flagged true."""
        pg.display.quit()
        pg.quit()
        if total: sys.exit()

    def _generate_grid(self, wall_pct=0.5):
        """
        Generate 100 grids sprinkled by wall_pct argument density.
        Raise error after 100 iterations of unsolvable grids,
        otherwise returns nparray grid and starting position tuple
        """

        for _ in range(100):
            # Initialize randomly sized grid
            min_x, max_x, min_y, max_y = self.map_size
            self.W = random.randint(min_x, max_x)
            self.H = random.randint(min_y, max_y)

            grid = np.zeros((NUM_ENTITIES, self.H, self.W), dtype=int)

            # Sprinkle in walls 
            for x in range(self.W):
                for y in range(self.H):
                    grid[WALL, y, x] = int(random.random() < self.wall_pct)

            # Insert start and goal TODO what does paper do?
            start, goal = self._random_tile(2, grid[WALL])
            grid[AGENT][start.p] = 1
            grid[GOAL][goal.p] = 1

            assert (not (grid[WALL][goal.p] or grid[WALL][start.p])), 'You done goofed'

            # If solvable, return
            if path := self._AStar(grid, start, goal):
                return grid, start, goal, path

        raise RuntimeError("Failed to create map after 100 tries! Your map"
                           "size is probably too small")

    def _random_tile(self, n=1, walls=[]):
        """Return random unique Vec2D tile"""
        rand = lambda: Vec2D(random.randint(0, self.W - 1),
                             random.randint(0, self.H - 1))  # random.randint is inclusive!!!
        rs = set()
        while len(rs) != n:
            tile = rand()
            if not walls[tile.p]:
                rs.add(tile)
            
        return list(rs)
    
    def _one_hot_encode(self, grid):
        """One hot encode grid"""
        shape = (grid.max(),) + grid.shape
        hot = np.zeros(shape)

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i, j]:  # Skip zeroes
                    hot[grid[i, j]-1, i, j] = 1
        return hot


    def _get_neighbors(self, n, grid):
        """ Return neigboring reachable Vec2D list of nodes to node Vec2D n"""
        neighbors = [n + d for d in self.DIRS]
        filtered = list(filter(lambda neighbor: not self._is_collide(neighbor, grid), neighbors))
        return filtered

    def _AStar(self, grid, start, goal):
        """AStar to quickly solve grid. 'n' refers to a node in the maze.  
            @returns: Vec2D list of shortest path or None if unsolvable
        """
        h = lambda n: max(abs(goal - n))  # Heuristic function "bird flight with diagonal movement
        # h = lambda n: abs(goal - n).sum  # Cartesian movement (proper planning)

        open_set = {start}  # Unvisited nodes
        came_from = {}  # Path tracking

        g_score = defaultdict(lambda: sys.maxsize)  # Cost of reaching n from start
        g_score[start] = 0

        f_score = defaultdict(lambda: sys.maxsize)  # g_score[n] + h(n)
        f_score[start] = h(start)

        while open_set:
            current = min(open_set, key=f_score.get)  # minimal f_score of n in open_set, i.e. best guess
            if current == goal:
                return self._reconstruct_path(came_from, current)  # Shortest path

            open_set.discard(current)
            for neighbor in self._get_neighbors(current, grid):
                tentative_gscore = g_score[current] + 1  # gScore[current] + d(current, neighbor) (d = 1)
                if tentative_gscore < g_score[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_gscore
                    f_score[neighbor] = tentative_gscore + h(neighbor)
                    if not neighbor in open_set:
                        open_set.add(neighbor)
        return None

    def _reconstruct_path(sekf, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def _is_collide(self, new_pos, grid):
        """Returns if new_pos Vec2D is out of bounds of nparray grid or colliding with wall"""
        return new_pos.x < 0 or new_pos.x >= self.W or new_pos.y < 0 or \
               new_pos.y >= self.H or grid[WALL][new_pos.p]

    def visit(self, action, pos): # ValueIteration helper function
        """Return tuple new pos and float reward from taking int action at tuple pos (x, y)"""
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        pos = Vec2D(pos)

        reward = self.step_penalty
        new_pos = pos + self.DIRS[action]

        if not self._is_collide(new_pos, self.grid):
            pos = new_pos
            if self.grid[GOAL][new_pos.p]:
                reward = self.win_reward

        return pos.p, reward

    def process_input(self):
        """Process user input quit/restart/step/space"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close(True)

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.close(True)

                if event.key == pg.K_r:
                    return self.reset()

                if event.key in MOVE:
                    return self.step(MOVE.index(event.key))

                if event.key == pg.K_SPACE:
                    try:
                        self.space_fun(self)
                    except:
                        self.space_fun()  # For external functions

    # ---- Value iteration ----
    def display_values(self, V):
        self.display.update_values(V)

    def set_pos(self, x, y):
        self.grid[AGENT][self.pos.p] = 0
        self.pos = Vec2D(x, y)
        self.grid[AGENT][self.pos.p] = 1


if __name__ == "__main__":
    env = GridWorld(seed=9, wall_pct=0.5, map_size=(5, 10, 5, 10), render=True, space_fun=GridWorld.test)
    env.reset()
    while True:
        obs = env.process_input()
        if type(obs) == tuple:  # step return
            s, r, done = obs
            if done:
                s = env.reset()
        elif type(obs) == np.ndarray:  # reset return
            grid = obs
