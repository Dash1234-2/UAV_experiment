import numpy as np
import math
from core import World, UAV, Entity

# 全局参数 这里暂且设置可偏转角度为90°，即在飞行高度为100m的情况下，无人机投影和可服务的地面用户的水平最大距离为100m
epsilon = 1e-3
p_0 = 99.66 # 叶片廓形功率, 单位w
p_1 = 120.16 # 推动功率
u_tip = 120 # 叶片转速， 单位m/s
v_0 = 4.03 # 悬停时平均旋翼诱导速度, 单位m/s
d_0 = 0.6 # 机身阻力比
s = 0.05 # 旋翼实度
p = 1.225 # 空气密度，单位kg/m³
T = 800  # 总执行时长，代表多少个timeslot
t = 4  # 每个时隙为4s
Energy = 2e8  # 无人机初始总能量，单位J 可供无人机进行足够的飞行和悬停
V = 20  # 无人机固定飞行速度，单位m/s
Range = 600
A = 0.503 # 旋翼圆盘面积


class Scenario:
    def make_world(self):
        world = World()
        world.num_UAVs = 2
        # world.num_landmarks = 5  # 50
        world.UAVs = [UAV() for i in range(world.num_UAVs)]
        # world.association = []
        # world.probability_LoS = 1 / (1 + A)
        for i, uav in enumerate(world.UAVs):
            uav.name = 'UAV %d'.format(i)
            uav.id = i
            uav.size = 10  
            uav.state.energy = Energy # 无人机的能量
        # # 列表形式的landmarks
        # world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d'.format(i)
        #     landmark.id = i
        #     landmark.size = 10
        # 字典形式的landmarks
        self.reset_world(world)  # 这里不reset会导致MultiUAVEnv中init中获得observation维度报错
        return world

    def reset_world(self, world):
        # world时隙初始化
        world.t = 0
        # 位置初始化,设置随机参数
        np.random.seed(666)
        np.random.seed(None)  # 取消随机种子
        # 位置初始化
        for uav in world.UAVs:
            uav.state.pos = np.array([Range / 2, Range / 2])
            # uav.state.pos = np.random.uniform(0, Range/4, 2)
            # uav.state.pos = np.random.uniform(0, Range, world.dim_p)
        # 能耗初始化
        for uav in world.UAVs:
            uav.state.energy = Energy

    # 奖励函数
    def reward(self, world):
        reward_list = self.get_sum_energy(world)
        # reward_list = np.expand_dims(reward_list, axis=1)
        # reward_list.shape:(2,)
        # print('reward_list.shape:{}'.format(np.array(reward_list).shape))
        Jain_index = self.get_Jain_index(world) 
        
        return Jain_index, reward_list

    def get_sum_energy(self, world):
        reward_list = []
        for i, uav in enumerate(world.UAVs):
            reward = 0           
            reward+=self.get_energy(uav)
            reward_list.append(-1*reward)
        return reward_list
    
    def get_energy(self, uav):
            distance_x = uav.action.distance_x * uav.max_distance
            distance_y = uav.action.distance_y * uav.max_distance
            distance = math.sqrt(distance_x*distance_x + distance_y* distance_y)
            # d = open('uav distance 日志.txt', 'a+')
            # print('uav distance:{}'.format(distance), file=d)
            fly_time = distance/V
            hover_time = t - fly_time
            # l = open('uav time 日志0.txt', 'a+')
            # print('fly_time:{0}, hover time:{1}'.format(fly_time, hover_time), file=l)
            # 单位悬停功耗
            energy_h = (p_0 + p_1)
            # 单位飞行功耗
            energy_blade = (p_0*(1+3*V*V/(u_tip*u_tip)))
            energy_induced = p_1 *math.sqrt((math.sqrt(1 + (math.pow(V, 4))/(4*math.pow(v_0, 4)))) - (V*V)/(2*v_0*v_0))
            energy_parasite = 0.5*d_0*p*s*A*(math.pow(V, 3))
            energy_f = (energy_parasite + energy_induced + energy_blade)
            total_energy = ((energy_f*fly_time) + (energy_h * hover_time))/1000
            return total_energy

    # 计算Jain系数
    def get_Jain_index(self, world):
        volumns = []
        for uav in world.UAVs:
            volumns.append(uav.sum_energy)
        volumns = np.array(volumns)
        Jain = np.power(np.sum(volumns), 2) / (len(volumns) * np.sum(np.power(volumns, 2)))
        return Jain


    def observation(self, world, uav):
        # 覆盖范围/观测范围
        obs_position = [uav.state.pos / Range]  # 未归一化
        for uav_tmp in world.UAVs:
            if uav is uav_tmp:
                continue
            else:
                obs_position.append((uav_tmp.state.pos - uav.state.pos) / Range)
            # obs_position.append(uav.state.pos)
        # obs_position.shape:(2, 2)
        # print('obs_position.shape:{}'.format(np.array(obs_position).shape))
        # obs_weight_norm.shape:(30,)
        # print('obs_weight_norm.shape:{}'.format(np.array(obs_weight_norm).shape))
        return np.concatenate(obs_position)

    def step(self, world):
        # 标致位，用来判断UAV此次运动是否越界
        is_out_bound = [False for _ in range(world.num_UAVs)]
        # 时隙自增
        world.t += 1
        # 更新位置和能耗
        j = 0
        for i, uav in enumerate(world.UAVs):
            distance_x = uav.action.distance_x * uav.max_distance
            distance_y = uav.action.distance_y * uav.max_distance
            uav.state.pos[0] += distance_x
            uav.state.pos[1] += distance_y
            uav.sum_energy+=self.get_energy(uav)
            uav.state.energy-=self.get_energy(uav)
            if uav.state.pos[0] < 0 or uav.state.pos[0] > Range or uav.state.pos[1] < 0 or uav.state.pos[1] > Range:
                is_out_bound[i] = True
                # continue
                uav.state.pos[0] -=distance_x
                uav.state.pos[1] -= distance_y
                j+=1
                world.t-=1
                if j>=2:
                    world.t+=(j-1)
                
        return is_out_bound

    def get_done(self, world):
        done = []
        for uav in world.UAVs:
            if uav.state.pos[0] < 0 or uav.state.pos[0] > Range or uav.state.pos[1] < 0 or uav.state.pos[1] > Range:
                done.append(True)
            else:
                done.append(False)
        return done


    def greedy_step(self, world):
        # 初始化
        # 标致位，用来判断UAV此次运动是否越界
        is_out_bound = [False for _ in range(world.num_UAVs)]
        # 时隙自增
        world.t += 1
        action_space_x = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        action_space_y = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        reward_list = []
        for uav in world.UAVs:
            pos_x = uav.state.pos[0]
            pos_y = uav.state.pos[1]
            action_x = 0
            action_y = 0
            energy = 0
            for i in action_space_x:
                for j in action_space_y:
                    total_energy = 0
                    pos_x = pos_x + i * uav.max_distance
                    pos_y = pos_y + j * uav.max_distance
                    if pos_x < 0 or pos_x > Range or pos_y < 0 or pos_y > Range:
                        pos_x = pos_x - i
                        pos_y = pos_y - j
                        break
                    # 计算当前情况系UAV的能耗
                    distance = math.sqrt(pos_x*pos_x + pos_y* pos_y)
                    fly_time = distance/V
                    hover_time = t - fly_time
                    # 单位悬停功耗
                    energy_h = (p_0 + p_1)
                    # 单位飞行功耗
                    energy_blade = (p_0*(1+3*V*V/(u_tip*u_tip)))
                    energy_induced = p_1 *math.sqrt(math.sqrt(1 + (math.pow(V, 4))/(4*math.pow(v_0, 4))) - (V*V)/(2*v_0*v_0))
                    energy_parasite = 0.5*d_0*p*s*A*(math.pow(V, 3))
                    energy_f = (energy_parasite + energy_induced + energy_blade)
                    total_energy = (energy_f*fly_time) + (energy_h * hover_time)
                    uav.sum_energy+=total_energy
                    if total_energy>energy:
                        energy = total_energy
                        action_x = i
                        action_y = j
                    ## 重置
                    pos_x = pos_x - i * uav.max_distance
                    pos_y = pos_y - j * uav.max_distance
                    

            print(action_x, action_y)

            # 更新该UAV的位置和关联用户
            reward_list.append(-1*total_energy)
            uav.state.pos += np.array([action_x*uav.max_distance, action_y*uav.max_distance])
        
            # uav_pos = np.array([pos_x, pos_y])
            #         # world.landmarks[str(landmark.id)].sum_throughput += self.get_capacity(uav, landmark)
            #         uav.associator.append(landmark.id)

        # return capacity_list




if __name__ == '__main__':
    # make_print_to_file()
    sc = Scenario()
    a = np.array([816.21, 531.57])
    b = np.array([752.02, 523.63])
    probability = sc.get_probability(a, b)
    print(probability)
