# from turtle import Vec2D
from logging import exception
import numpy as np
import pygame as pg
import sys, random

CELL = 48 # Cell width TODO: Makes dependent on map_size
WHITE, GREY, BLACK = (240, 240, 240),  (200, 200, 200), (20, 20, 20)
YELLOW, RED, GREEN = (200, 200, 0), (200, 0, 0), (0, 200, 0)
COLORS = [BLACK, YELLOW, GREEN]  # empty, wall, goal

MOVE = [pg.K_d, pg.K_e, pg.K_w, pg.K_q,  #RIGHT, RIGHT_UP, UP, LEFT-UP
        pg.K_a, pg.K_z, pg.K_s, pg.K_c]  #LEFT, LEFT-DOWN, DOWN, RIGHT-DOWN 

pg.init()
pg.display.set_caption('GridWorld')
# pg.font.init()

class Vec2D():
    # Helper struct for flipping coordinates and basic arithmetic.
    # pygame draws by (x, y) and numpy indexes by (y, x).

    def __init__(self, x, y=0):
        if type(x) != tuple:
            self.x, self.y = x, y
            self.v = x, y  # pygame rect
            self.p = y, x  # numpy array
        else:
            self.v = x
            self.x, self.y = x[0], x[1]
            self.p = self.y, self.x

    def __repr__(self):
        return str((self.x, self.y))

    def __add__(self, o):
        return Vec2D(self.x + o.x, self.y + o.y)

    def __mul__(self, k):
        return Vec2D(k*self.x, k*self.y)

    def __eq__(self, o):
        return self.x == o.x and self.y == o.y

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

    def __init__(self, map_size=(5, 10, 5, 10), wall_pct=0.2, render=True):
        self.map_size = map_size
        self.wall_pct = wall_pct
        self.render = render
        self.directions = [Vec2D(1, 0), Vec2D(1,-1), # RIGHT, RIGHT-UP
                           Vec2D(0,-1),Vec2D(-1,-1), # UP, LEFT-UP
                           Vec2D(-1,0), Vec2D(-1,1), # LEFT, LEFT-DOWN
                           Vec2D(0, 1), Vec2D(1, 1)] # DOWN, RIGHT-DOWN
        self.reset()
    
    def step(self, action):
        new_pos = self.pos + self.directions[action]
        
        terminate = self._is_collide(new_pos)
        if not terminate:
            if self.render:
                self.screen.update(new_pos, self.pos)

            self.pos = new_pos
    
    def reset(self):
        # Generate new grid
        # {0:empty, 1:wall, 3:end}
        # self.done = False

        self.grid, start = self._generate_grid()
        self.pos = Vec2D(start)
        # self.grid = np.zeros((self.H, self.W), dtype=int)
        # start = (0, 0)
        # self.pos = Vec2D(start)
        # self.goal = Vec2D(self.W-1, self.H-1)
        # self.grid[self.goal.p] = 2
        # self.grid[0, 4] = 1
        # self.grid[1, 3] = 1
        # self.grid[2, 2] = 1
        # self.grid[1, 1] = 1

        self.screen = Display(self.grid, self.pos)
        
    def _generate_grid(self):
        # Tries 100 times to solve a randomly generated grid using wall percentage
        # @ Returns: (nparray, tuple) a randomly generated grid and starting position

        for _ in range(100):
            # Initialize randomly sized grid
            min_x, max_x, min_y, max_y = self.map_size
            self.W = random.randint(min_x, max_x)
            self.H = random.randint(min_y, max_y)
            grid = np.empty((self.H, self.W), dtype=int)

            # Sprinkle in walls https://github.com/facebookarchive/MazeBase/blob/master/py/mazebase/utils/creationutils.py
            for x in range(self.W):
                for y in range(self.H):
                    grid[y, x] = int(random.random() < self.wall_pct)
            
            # Insert start and goal TODO what does paper do?
            start, goal = self._random_tile(2)
            self.pos = Vec2D(start)
            grid[start] = 0
            grid[goal] = 2

            # If solvable, return
            if self._dijkstra(grid):
                return grid, start
        
        raise RuntimeError("Failed to create map after 100 tries! Your map"
	               "size is probably too small")
    
    def _random_tile(self, n=1):
        return [(random.randint(0, self.H-1), random.randint(0, self.W-1)) for _ in range(n)]  # random.randint is inclusive!!!
    
    def _dijkstra(self, grid):
        jason_cool = True
        jason_no_gf = True
        thats_fucked_up = jason_cool and jason_no_gf
        return thats_fucked_up


    def _is_collide(self, new_pos):
        # Out of bounds or collide with wall
        return new_pos.x < 0 or new_pos.x >= self.W or new_pos.y < 0 or \
               new_pos.y >= self.H or self.grid[new_pos.p] == 1  # 1:Wall
    
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
                    


GW = GridWorld()

while True:
    GW.process_input()


