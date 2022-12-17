import numpy as np

Range = 600

class State:
    def __init__(self):
        self.pos = None



class UAVState(State):
    def __init__(self):
        super(UAVState, self).__init__()
        self.energy = None



class Action:
    def __init__(self):
        self.distance_x = None
        self.distance_y = None


class Entity:
    def __init__(self):
        self.name = ''
        self.type = None
        self.id = None
        self.size = 5
        self.movable = False
        self.color = None
        self.state = None



class UAV(Entity):
    def __init__(self):
        super(UAV, self).__init__()
        self.movable = True
        self.state = UAVState()
        self.action = Action()
        self.height = 50
        self.max_distance = 40 # 80
        self.coverage = 100
        self.sum_energy = 0


class World:
    def __init__(self):
        self.UAVs = []
        self.dim_p = 2
        self.dt = 0.1  # simulation timestep
        self.length = Range
        self.width = Range
        self.t = 0 # 时隙
        self.num_UAVs = 0
        self.num_landmarks = 0

    # @property  # 装饰器，将函数改为可以直接调用的变量
    # def entities(self):
    #     return self.UAVs + self.landmarks

