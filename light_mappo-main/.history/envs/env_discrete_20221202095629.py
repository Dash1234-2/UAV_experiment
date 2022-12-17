"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : env_discrete.py
"""

import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore
from envs.multiUAV_sceneario import Scenario as sc
Range = 600

class DiscreteActionEnv(object):
    """对于动作环境的封装"""
    def __init__(self):
        # self.env = EnvCore()
        self.env = sc()
        world = self.env.make_world()
        self.world = world
        self.num_agent = world.num_UAVs
        # for uav in self.world.UAVs:
        #     self.signal_obs_dim = len(self.env.observation(self.world, uav))
        self.signal_action_dim = 2

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        for agent in self.world.UAVs:
            total_action_space = []

            # physical action space
            # u_action_space = spaces.Discrete(self.signal_action_dim)  # 5个离散的动作
            u_action_space = spaces.Box(low=-1, high=1, shape=(self.signal_action_dim,), dtype=np.float32)

            if self.movable:
                total_action_space.append(u_action_space)

            self.action_space.append(total_action_space[0])
                # self.action_space.append(total_action_space[0])
               

            # observation space
            obs_dim = len(self.env.observation(self.world, agent))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]
                                                
        self.viewer = None
        self._reset_render()
    def step(self, actions):
        """
        输入actions纬度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码
        """

        # results = self.env.step(actions)
        # obs, rews, dones, infos = results
        # infos = {}
        obs_n = []
        reward = 0
        # 将action更新到每个UAV agent的属性中去
        for i, uav in enumerate(self.world.UAVs):
            self._set_action(actions[i], uav, self.action_space[i])
        # 更新状态
        is_out_bound = self.env.step(self.world)
        is_out_bound_array = []
        for tmp in is_out_bound:
            if tmp:
                is_out_bound_array.append(-1)
            else:
                is_out_bound_array.append(1)
        is_out_bound_array = np.array(is_out_bound_array)
        # 记录观察
        for uav in self.world.UAVs:
            obs_n.append(self._get_obs(uav))
        # 查看当前环境是否因越界或能量耗尽而结束
        done_n = self.env.get_done(self.world)
        infos, reward = self.env.reward(self.world)
        reward = np.multiply(is_out_bound_array, np.array(reward))
        return np.stack(obs_n), np.stack(reward), np.stack(done_n), infos

    def _set_action(self, action, uav, action_space, time=None):
        uav.action.distance_x = action[0]
        uav.action.distance_y = action[1]

    def _get_obs(self, uav):
        return self.env.observation(self.world, uav)

    def _get_reward(self, uav):
        return self.env.reward(self.world, uav)

    def _get_done(self, uav):
        return self.env.reward(self.world, uav)

    def reset(self):
        self.env.reset_world(self.world)
        self._reset_render()
        obs_n = []
        for uav in self.world.UAVs:
            obs_n.append(self.env.observation(self.world, uav))
        return np.stack(obs_n)

    def _reset_render(self):
        self.render_geom = None
        self.render_transform = None

    def random_action(self):
        action_n = []  # 随机联合动作
        for action in self.action_space:
            action_n.append(action.sample())
        return action_n

    def get_Jain_Index(self):
        ans = self.sc.get_Jain_index(self.world)
        return ans

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        # 新代码 不适用tranform，直接刷新全局组件
        screen_width = Range
        screen_height = Range
        # 如果没有viewer，创建viewer和uav、landmarks
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_height, screen_width)
        self.viewer.set_bounds(0, Range, 0, Range)
        self.viewer.geoms.clear()
        for uav in self.world.UAVs:
            geom = rendering.make_circle(uav.size)
            geom.set_color(1, 0, 0)
            geom_form = rendering.Transform(translation=(uav.state.pos[0], uav.state.pos[1]))
            geom.add_attr(geom_form)
            self.viewer.add_geom(geom)
        for landmark in self.world.landmarks.values():
            geom = rendering.make_circle(landmark.size)
            geom.set_color(0, 1, 0)
            geom_transform = rendering.Transform(translation=(landmark.state.pos[0], landmark.state.pos[1]))
            geom.add_attr(geom_transform)
            self.viewer.add_geom(geom)
        for uav in self.world.UAVs:
            for i in uav.associator:
                line = rendering.Line(uav.state.pos, self.world.landmarks[str(i)].state.pos)
                self.viewer.add_geom(line)

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    DiscreteActionEnv().step(actions=None)