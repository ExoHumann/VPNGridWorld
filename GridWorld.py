import numpy as np
import pygame as pg
import sys

WHITE, GREY, BLACK = (240, 240, 240),  (200, 200, 200), (20, 20, 20)
YELLOW, RED = (200, 200, 0), (200, 0, 0) 

pg.init()
pg.display.set_caption('GridWorld')
# pg.font.init()

class Screen():

    def __init__(self, dim):
        self.size = 24
        self.W, self.H = dim
        self.screen = pg.display.set_mode([self.W * self.size, self.H * self.size])
        self.font = pg.font.Font(None, 25)
        self.reset()
        self.update(Vector2D((0,0)), Vector2D((0,0)))
    
    def update(self, pos, before):
        x, y = before.p
        self._draw_rect(x, y, BLACK)
        self._draw_circle(pos)
    
    def reset(self):
        self.screen.fill(WHITE)
        pg.display.flip()

        for x in range(self.W):
            for y in range(self.H):
                self._draw_rect(x, y, BLACK)
    
    def _draw_rect(self, x, y, color):
        pos = Vector2D((x, y)) * self.size + Vector2D((1, 1))
        rect = pg.Rect(pos.x, pos.y, self.size-2, self.size-2)
        pg.draw.rect(self.screen, color, rect, 0)
        pg.display.update(rect)
    
    def _draw_circle(self, pos):
        center = (pos + Vector2D((-0.5, -0.5))) * self.size
        pg.draw.circle(self.screen, WHITE, center, 20)


class Vector2D():

	def __init__(self, point):
		self.x, self.y = point
		self.p = self.y, self.x  # State indexing

	def __repr__(self):
		return str((self.x, self.y))

	def __add__(self, o):
		return Vector2D((self.x + o.x, self.y + o.y))

	def __mul__(self, k):
		return Vector2D((k*self.x, k*self.y))

	# def __rmul__(self, o):
	# 	return self.x*o.x + self.y*o.y

	def __eq__(self, o):
		return self.x == o.x and self.y == o.y

class GridWorld():

    def __init__(self, size=(5, 5), start=(0,0), end=None, render=False):
        self.W, self.H = size
        self.pos = Vector2D(start)
        self.end = Vector2D(size)
        self.render = render

        self.board = np.zeros((size), dtype=int)

        # self.directions = [Vector2D((1, 0)), Vector2D((1, -1)), Vector2D((0, -1)),  # RIGHT, RIGHT-UP, UP
		# 				   Vector2D((-1, -1)), Vector2D((-1, 0)), Vector2D((-1, 1)), # LEFT-UP, LEFT, LEFT-DOWN
        #                    Vector2D((0, 1)), Vector2D((1, 1))]  # DOWN, RIGHT-DOWN
        self.directions = [Vector2D((1, 0)), Vector2D((0, -1)),  # RIGHT, UP
						   Vector2D((-1, 0)), Vector2D((0, 1))]  # LEFT, DOWN
        
        self.screen = Screen(size)
        
    def step(self, action):
        new_pos = self.pos + self.directions[action]
        terminate = new_pos.x < 0 or new_pos.x >= self.W or new_pos.y < 0 or new_pos.y >= self.H

        if not terminate:
            self.pos = new_pos

            # if self.render:

    def process_input(self):
		# event = pg.event.wait()
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
                if event.key == pg.K_SPACE:
                    self.rendering = not self.rendering
                    if self.rendering:
                        self.screen.reset(self, self.fruit)

                    if event.key in MOVE:
                        self.step(MOVE.index(event.key))

MOVE = [pg.K_RIGHT, pg.K_UP, pg.K_LEFT, pg.K_DOWN]

GW = GridWorld()

while True:
    GW.process_input()


