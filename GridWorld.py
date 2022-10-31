from turtle import Vec2D
import numpy as np
import pygame as pg
import sys

WHITE, GREY, BLACK = (240, 240, 240),  (200, 200, 200), (20, 20, 20)
YELLOW, RED = (200, 200, 0), (200, 0, 0) 

pg.init()
pg.display.set_caption('GridWorld')
# pg.font.init()

class Screen():

    def __init__(self, board):
        self.size = 24
        self.W, self.H = board.shape
        self.screen = pg.display.set_mode([self.W * self.size, self.H * self.size])
        self.board = board
        # self.font = pg.font.Font(None, 25)
        self.reset()
        self.update(Vector2D((0,0)), Vector2D((0,0)))

    def update(self, pos, before):
        x, y = before.p
        self._draw_rect(before, BLACK)
        self._draw_circle(pos)
    
    def reset(self):
        self.screen.fill(WHITE)
        pg.display.flip()

        for x in range(self.W):
            for y in range(self.H):
                pos = Vector2D((x, y))
                color = YELLOW if self.board[pos.p] else BLACK
                self._draw_rect(pos, color)

    def _draw_rect(self, pos, color):
        pos = pos * self.size + Vector2D((1, 1))
        rect = pg.Rect(pos.v, (self.size-2, self.size-2))
        pg.draw.rect(self.screen, color, rect, 0)
        pg.display.update(rect)
    
    def _draw_circle(self, pos):
        center = (pos + Vector2D((0.5, 0.5))) * self.size
        rect = pg.draw.circle(self.screen, WHITE, center.v, 8)
        pg.display.update(rect)


class Vector2D():
    def __init__(self, point):
        self.v = point           # pygame rect
        self.x, self.y = point
        self.p = self.y, self.x  # numpy array

    def __repr__(self):
        return str((self.x, self.y))

    def __add__(self, o):
        return Vector2D((self.x + o.x, self.y + o.y))

    def __mul__(self, k):
        return Vector2D((k*self.x, k*self.y))

    def __eq__(self, o):
        return self.x == o.x and self.y == o.y

class GridWorld():
    """ Gridworld environment
    8 actions
    board {0 : empty, 1 : agent, 2 : wall, 3 : start, 4 : end}
    
    """

    def __init__(self, dim=(5, 5), start=(0,0), end=None, render=True):
        self.dim = dim
        self.W, self.H = dim
        self.render = render
        self.directions = [Vector2D((1, 0)), Vector2D((1, -1)), Vector2D((0, -1)),  # RIGHT, RIGHT-UP, UP
	    				   Vector2D((-1, -1)), Vector2D((-1, 0)), Vector2D((-1, 1)), # LEFT-UP, LEFT, LEFT-DOWN
                           Vector2D((0, 1)), Vector2D((1, 1))]  # DOWN, RIGHT-DOWN
        #self.directions = [Vector2D((1, 0)), Vector2D((0, -1)),  # RIGHT, UP
		# 				   Vector2D((-1, 0)), Vector2D((0, 1))]  # LEFT, DOWN
        
        self.reset()
    
        
    def step(self, action):
        new_pos = self.pos + self.directions[action]
        
        terminate = self._is_collide(new_pos)
        if not terminate:
            if self.render:
                self.screen.update(new_pos, self.pos)

            self.pos = new_pos
    
    def reset(self):
        # Generate new board
        # self.done = False
        self.board = np.zeros(self.dim, dtype=int)

        self.pos = Vector2D((0,0))
        self.end = Vector2D((self.dim))
        self.board[0, 4] = 1
        self.board[1, 3] = 1

        self.screen = Screen(self.board)
        

    def _is_collide(self, new_pos):
        return new_pos.x < 0 or new_pos.x >= self.W or new_pos.y < 0 or \
               new_pos.y >= self.H or self.board[new_pos.p]


    def process_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.display.quit()
                pg.quit()
                sys.exit()
            
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.display.quit()
                    pg.quit()
                    sys.exit()

                if event.key == pg.K_r:
                    self.reset()

                if event.key in MOVE:
                    self.step(MOVE.index(event.key))
                    
MOVE = [pg.K_d, #RIGHT
        pg.K_e,               #RIGHT_UP
        pg.K_w,    #UP
        pg.K_q,               #LEFT-UP
        pg.K_a,  #LEFT
        pg.K_z,               #LEFT-DOWN
        pg.K_s,  #DOWN
        pg.K_c]               #RRIGHT-DOWN

GW = GridWorld()

while True:
    GW.process_input()


