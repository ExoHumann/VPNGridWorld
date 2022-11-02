""" GridWorld environment for RL
    Run this script for human play

    Common bugs
    -  "list index out of range" => random.randint(a, b) is inclusive [a, b] as opposed to exlusive [a, b) as in numpy

"""


from logging import exception
from gym import spaces
import numpy as np
import pygame as pg
import sys, random
from collections import defaultdict

CELL = 48 # Cell width TODO: Makes dependent on map_size but still global...
WHITE, GREY, BLACK = (200, 200, 200),  (190, 190, 190), (20, 20, 20)
YELLOW, RED, GREEN = (150, 150, 0), (200, 0, 0), (0, 150, 0)
COLORS = [BLACK, YELLOW, WHITE, GREEN]  # empty, wall, goal

MOVE = [pg.K_d, pg.K_e, pg.K_w, pg.K_q,  #RIGHT, RIGHT_UP, UP, LEFT-UP
        pg.K_a, pg.K_z, pg.K_s, pg.K_c]  #LEFT, LEFT-DOWN, DOWN, RIGHT-DOWN 
EMPTY, WALL, AGENT, GOAL = range(4)

# pg.init()
pg.display.set_caption('GridWorld')
# pg.font.init()

class Vec2D():
    # Helper struct for flipping coordinates and basic arithmetic.
    # pygame draws by (x, y) and numpy indexes by (y, x).

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
        return Vec2D(k*self.x, k*self.y)
    
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
    # Handles all rendering

    def __init__(self, grid, start, path=None):
        self.grid = grid
        self.H, self.W = grid.shape
        self.path = path
        self.screen = pg.display.set_mode([self.W * CELL, self.H * CELL])
        # self.font = pg.font.Font(None, 25)
        self.reset(start)

    def update(self, pos, before):
        self._draw_rect(before, COLORS[self.grid[before.p]])    # Draw prior object
        if self.path and before in self.path: # Draw path point
            self._draw_circle(before, CELL >> 3)
        self._draw_circle(pos, CELL >> 2)  # x >> 2 = x / 4     # Draw agent
    
    def reset(self, start):
        self.screen.fill(GREY)
        pg.display.flip()

        for x in range(self.W):
            for y in range(self.H):
                pos = Vec2D(x, y)
                color = COLORS[self.grid[pos.p]]
                self._draw_rect(pos, color)
        self.draw_path()
        self._draw_circle(start, CELL >> 2)  # Starting pos
    
    def draw_path(self):
        for p in self.path:
            self._draw_circle(p, CELL >> 3)  # x >> 2 = x / 8

    def _draw_rect(self, pos, color):
        pos = pos * CELL + Vec2D(1, 1)
        rect = pg.Rect(pos.v, (CELL-2, CELL-2))
        pg.draw.rect(self.screen, color, rect, 0)
        pg.display.update(rect)
    
    def _draw_circle(self, pos, r):
        center = (pos + Vec2D(0.5, 0.5)) * CELL
        rect = pg.draw.circle(self.screen, GREY, center.v, r)  
        pg.display.update(rect)
    
    

class GridWorld():
    """ Gridworld environment
    8 actions {0:Right, 1:RIGHT-UP, 2:UP, 3:LEFT-UP, 4:LEFT, 
              5:LEFT-DOWN, 6:DOWN, 7:RIGHT-DOWN}
    grid {0:Empty, 1:Wall, 2:Agent, 3:Goal)
    """

    def test(self):
        print(self.observation_space)

    def __init__(self, map_size=(5, 10, 5, 10), wall_pct=0.7, render=True, seed=None, space_fun=None):
        self.map_size = map_size
        self.wall_pct = wall_pct
        self.render = render
        if seed != None:
            random.seed(seed)
        self.space_fun = space_fun if space_fun else lambda: None

        self.DIRS = [Vec2D(1, 0), Vec2D(1,-1), # RIGHT, RIGHT-UP
                     Vec2D(0,-1),Vec2D(-1,-1), # UP, LEFT-UP
                     Vec2D(-1,0), Vec2D(-1,1), # LEFT, LEFT-DOWN
                     Vec2D(0, 1), Vec2D(1, 1)] # DOWN, RIGHT-DOWN
        
        self.action_space = spaces.Discrete(len(self.DIRS))
        # observation_space defined in reset()
        
        self.reset()
    
    def reset(self):
        # Generate new grid
        # {0:empty, 1:wall, 2:goal}
        # self.done = False
        self.grid, start, goal, path = self._generate_grid()
        self.observation_space = spaces.Box(0, 2, shape=self.grid.shape, dtype=int)
        self.pos = start
        self.goal = goal
            
        assert(self.grid[self.pos.p] == AGENT), f"Improper starting tile"
        if self.render:
            self.display = Display(self.grid, self.pos, path)

        return self.grid
    
    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        done = False
        reward = 0

        new_pos = self.pos + self.DIRS[action]
        
        terminate = self._is_collide(new_pos, self.grid)
        if not terminate:
            if self.grid[new_pos.p] == GOAL:
                # print("GOOOAL")
                done = True
            if self.render:
                self.display.update(new_pos, self.pos)

            self.pos = new_pos
        return self.grid, reward, done
    
    def sample(self, legal_only=False):
        # Todo: Use gyms action space instead
        if legal_only:
            legal_actions = list(filter(lambda dir: not self._is_collide(self.pos + dir, self.grid), self.DIRS))
            action = random.choice(legal_actions)
            return self.DIRS.index(action)
        return random.randint(0, len(self.DIRS)-1)
        
    def _generate_grid(self):
        # Tries 100 times to solve a randomly generated grid using wall percentage
        # Sprinkling: https://github.com/facebookarchive/MazeBase/blob/master/py/mazebase/utils/creationutils.py
        # @ Returns: (nparray, tuple) a randomly generated grid and starting position
        
        for _ in range(100):
            # Initialize randomly sized grid
            min_x, max_x, min_y, max_y = self.map_size
            self.W = random.randint(min_x, max_x)
            self.H = random.randint(min_y, max_y)
            grid = np.empty((self.H, self.W), dtype=int)

            # Sprinkle in walls 
            for x in range(self.W):
                for y in range(self.H):
                    grid[y, x] = int(random.random() < self.wall_pct)
            
            # Insert start and goal TODO what does paper do?
            start, goal = self._random_tile(2)
            grid[start.p] = AGENT
            grid[goal.p] = GOAL

            # If solvable, return
            if path := self._AStar(grid, start, goal):
                return grid, start, goal, path
        
        raise RuntimeError("Failed to create map after 100 tries! Your map"
	               "size is probably too small")
    
    def _random_tile(self, n=1):
        # Unique elements
        # returns random Vec2D(x, y) list
        rand = lambda: Vec2D(random.randint(0, self.W-1), random.randint(0, self.H-1))  # random.randint is inclusive!!!
        rs = {rand()}
        while len(rs) != n:
            rs.add(rand())
        return list(rs)  
    
    def get_neighbors(self, n, grid):
        """ return neigboring nodes
        @params: Vec2D of node,
                nparray of grid
        @returns: Vec2D list
        """
        neighbors = [n + d for d in self.DIRS]
        filtered = list(filter(lambda neighbor: not self._is_collide(neighbor, grid), neighbors))
        return filtered
    
    def _AStar(self, grid, start, goal):
        """AStart algoritm to quickly solve grid.
            Uncomment came_from to return path and/or set a distance function based on rewards (currently 1)
            @returns: Vec2D list of shortest path or None if unsolvable
        """
        h = lambda n: max(abs(goal - n))  # Heuristic function "bird flight with diagonal movement
        # h = lambda n: abs(goal - n).sum   # Cartesian movement

        open_set = {start}  # Unvisited nodes
        came_from = {}  # Path tracking

        g_score = defaultdict(lambda: sys.maxsize) # Cost of reaching n
        g_score[start] = 0

        f_score = defaultdict(lambda: sys.maxsize)  # g_score[n] + h(n)
        f_score[start] = h(start)

        while open_set:
            current = min(open_set, key=f_score.get)  # minimal f_score of n in open_set, i.e. best guess
            if current == goal:
                return self._reconstruct_path(came_from, current)  # Shortest path
        
            open_set.discard(current)
            for neighbor in self.get_neighbors(current, grid):
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
        # Out of bounds or collide with wall
        return new_pos.x < 0 or new_pos.x >= self.W or new_pos.y < 0 or \
               new_pos.y >= self.H or grid[new_pos.p] == WALL  # 1:Wall
    
    def close(self):
        pg.display.quit()
        pg.quit()
        sys.exit()

    def process_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
            
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.close()

                if event.key == pg.K_r:
                    self.reset()

                if event.key in MOVE:
                    return self.step(MOVE.index(event.key))
                
                if event.key == pg.K_SPACE:
                    self.space_fun(self)
                    
                    

if __name__ == "__main__":
    env = GridWorld(seed=42, wall_pct=0.5, space_fun=GridWorld.test)
    while True:
        obs = env.process_input()
        if obs:
            s, r, done = obs
            if done:
                env.reset()


