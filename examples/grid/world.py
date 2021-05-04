import numpy as np
import random
from collections import namedtuple

from base_agent.base_util import Pos, Look

Bot = namedtuple("Bot", "entityId, name, pos, look")
import matplotlib
matplotlib.use('TkAgg')

WORLD_SIZE = 5

class World:
    def __init__(self, opts=None, spec=None):
        self.bots = []
        for i in range(opts.num_bots):
            x = random.randint(0, WORLD_SIZE - 1)
            y = random.randint(0, WORLD_SIZE - 1)
            eid = i
            botname = "target" + str(i)
            bot = Bot(eid, botname, Pos(x, y, 0), Look(0, 0))
            self.bots.append(bot)
    
    def step(self):
        self.bots = [self.random_move(bot) for bot in self.bots]
    
    def get_bots(self, eid=None):
        bots = self.bots if eid is None else [b for b in self.bots if b.entityId == eid]
        return bots
    
    def remove_bot(self, eid):
        print(f"[World INFO]: Remove bot with entity id [{eid}] from the world.")
        self.bots[:] = [b for b in self.bots if b.entityId != eid]
    
    def random_move(self, bot):
        move_delta = []
        (x, y, z) = bot.pos
        if x > 0:
            move_delta.append((-1, 0, 0)) # allow move in negative x axis
        if x < 4:
            move_delta.append((1, 0, 0)) # allow move in positive x axis
        if y > 0:
            move_delta.append((0, -1, 0)) # allow move in negative y axis
        if y < 4:
            move_delta.append((0, 1, 0)) # allow move in positive y axis
        move = random.choice(move_delta)
        x, y, z = x + move[0], y + move[1], z + move[2]
        return bot._replace(pos=Pos(x, y, z))
    
    def visualize(self, agent):
        import matplotlib.pyplot as plt

        nrows, ncols = 5, 5
        grid = np.zeros((nrows, ncols))

        x, y, z = agent.get_pos()
        if len(self.bots) <= 0:
            return
        for b in self.bots:
            tx, ty, tz = b.pos
            grid[tx, ty] = b.entityId + 2
        grid[x, y] = 1
        plt.matshow(grid, fignum=1)

        plt.xticks(range(ncols))
        plt.yticks(range(nrows))
        plt.pause(0.05)
