""" GridWorld environment for RL
Common bugs
-  random.randint(a, b) is inclusive [a, b] as opposed to exlusive [a, b) as in numpy

"""


from logging import exception
import numpy as np
import pygame as pg
import sys, random
from collections import defaultdict

CELL = 48 # Cell width TODO: Makes dependent on map_size but still global...
WHITE, GREY, BLACK = (240, 240, 240),  (200, 200, 200), (20, 20, 20)
YELLOW, RED, GREEN = (200, 200, 0), (200, 0, 0), (0, 200, 0)
COLORS = [BLACK, YELLOW, GREEN]  # empty, wall, goal

MOVE = [pg.K_d, pg.K_e, pg.K_w, pg.K_q,  #RIGHT, RIGHT_UP, UP, LEFT-UP
        pg.K_a, pg.K_z, pg.K_s, pg.K_c]  #LEFT, LEFT-DOWN, DOWN, RIGHT-DOWN 

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

    def __init__(self, grid, start):
        self.grid = grid
        self.H, self.W = grid.shape
        self.screen = pg.display.set_mode([self.W * CELL, self.H * CELL])
        # self.font = pg.font.Font(None, 25)
        self.reset(start)

    def update(self, pos, before):
        self._draw_rect(before, COLORS[self.grid[before.p]])
        self._draw_agent(pos)
    
    def reset(self, start):
        self.screen.fill(WHITE)
        pg.display.flip()

        for x in range(self.W):
            for y in range(self.H):
                pos = Vec2D(x, y)
                color = COLORS[self.grid[pos.p]]
                self._draw_rect(pos, color)
        self._draw_agent(start)

    def _draw_rect(self, pos, color):
        pos = pos * CELL + Vec2D(1, 1)
        rect = pg.Rect(pos.v, (CELL-2, CELL-2))
        pg.draw.rect(self.screen, color, rect, 0)
        pg.display.update(rect)
    
    def _draw_agent(self, pos):
        center = (pos + Vec2D(0.5, 0.5)) * CELL
        rect = pg.draw.circle(self.screen, WHITE, center.v, CELL >> 2)  # x >> 2 = x / 4
        pg.display.update(rect)

class GridWorld():
    """ Gridworld environment
    8 actions {0:Right, 1:RIGHT-UP, 2:UP, 3:LEFT-UP, 4:LEFT, 
              5:LEFT-DOWN, 6:DOWN, 7:RIGHT-DOWN}
    grid {0:Empty, 1:Wall, 2:Goal]
    """

    def __init__(self, map_size=(5, 10, 5, 10), wall_pct=0.7, render=True, seed=None):
        self.map_size = map_size
        self.wall_pct = wall_pct
        self.render = render
        if not seed == None:
            random.seed(seed)

        self.DIRS = [Vec2D(1, 0), Vec2D(1,-1), # RIGHT, RIGHT-UP
                     Vec2D(0,-1),Vec2D(-1,-1), # UP, LEFT-UP
                     Vec2D(-1,0), Vec2D(-1,1), # LEFT, LEFT-DOWN
                     Vec2D(0, 1), Vec2D(1, 1)] # DOWN, RIGHT-DOWN
        self.reset()
    
    def step(self, action):
        new_pos = self.pos + self.DIRS[action]
        
        terminate = self._is_collide(new_pos, self.grid)
        if not terminate:
            if self.grid[new_pos.p] == 2:
                print("GOOOAL")
            if self.render:
                self.screen.update(new_pos, self.pos)

            self.pos = new_pos
    
    def reset(self):
        # Generate new grid
        # {0:empty, 1:wall, 2:goal}
        # self.done = False
        self.grid, start, goal = self._generate_grid()
        self.pos = start
        self.goal = goal
            
        assert(self.grid[self.pos.p] == 0), f"Improper starting tile"

        self.screen = Display(self.grid, self.pos)
        
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
            grid[start.p] = 0
            grid[goal.p] = 2

            # If solvable, return
            if self._AStar(grid, start, goal):
                return grid, start, goal
        
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
            @returns: bool on whether grid is solvable 
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
                print(self._reconstruct_path(came_from, current))  # Shortest path
                return True
        
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
               new_pos.y >= self.H or grid[new_pos.p] == 1  # 1:Wall
    
    def close(self):
        pg.display.quit()
        pg.quit()
        sys.exit()

    def process_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.display.quit()
                pg.quit()
                sys.exit()
            
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.close()

                if event.key == pg.K_r:
                    self.reset()

                if event.key in MOVE:
                    self.step(MOVE.index(event.key))
                    


GW = GridWorld(seed=42)
while True:
    GW.process_input()

