import numpy as np
import math
from env_core import World, UAV, Entity

# 全局参数 这里暂且设置可偏转角度为90°，即在飞行高度为100m的情况下，无人机投影和可服务的地面用户的水平最大距离为100m
epsilon = 1e-3
p_0 = 99.66 # 叶片廓形功率, 单位w
p_1 = 120.16 # 推动功率
u_tip = 120 # 叶片转速， 单位m/s
v_0 = 4.03 # 悬停时平均旋翼诱导速度, 单位m/s
d_0 = 0.6 # 机身阻力比
s = 0.05 # 旋翼实度
ρ = 1.225 # 空气密度，单位kg/m³

T = 1000  # 总执行时长，代表多少个timeslot,120个时隙为2400s。
t = 4  # 每个时隙为4s
alpha = 0.2  # 能耗所占比重
Energy = 2e7  # 无人机初始总能量，单位J 可供无人机进行足够的飞行和悬停
V = 20  # 无人机固定飞行速度，单位m/s
Range = 600
A = 0.503 # 旋翼圆盘面积



class Scenario:
    def make_world(self):
        world = World()
        world.num_UAVs = 2
        world.num_landmarks = 5  # 50
        world.UAVs = [UAV() for i in range(world.num_UAVs)]
        world.association = []
        world.probability_LoS = 1 / (1 + A)
        for i, uav in enumerate(world.UAVs):
            uav.name = 'UAV %d'.format(i)
            uav.id = i
            uav.size = 10  # size表示什么？
            uav.state.energy = Energy # 无人机的能量
        # # 列表形式的landmarks
        # world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d'.format(i)
        #     landmark.id = i
        #     landmark.size = 10
        # 字典形式的landmarks
        for i in range(world.num_landmarks):
            dic = {str(i): Landmark()}
            world.landmarks.update(dic)
        for i, landmark in enumerate(world.landmarks.values()):
            landmark.name = 'landmark %d'.format(i)
            landmark.id = i
            landmark.size = 5
        self.reset_world(world)  # 这里不reset会导致MultiUAVEnv中init中获得observation维度报错
        return world

    def reset_world(self, world):
        # world时隙初始化
        world.t = 0
        # 位置初始化,设置随机参数
        np.random.seed(666)
        landmarks_position = np.random.uniform(0, Range, (len(world.landmarks), 2))
        np.random.seed(None)  # 取消随机种子
        for i, landmark in enumerate(world.landmarks.values()):
            landmark.state.pos = landmarks_position[i]
            landmark.weight = 1
            landmark.sum_throughput = 1
            landmark.avg_dataRate = 0
        for uav in world.UAVs:
            uav.state.pos = np.array([Range / 2, Range / 2])
            # uav.state.pos = np.random.uniform(0, Range, world.dim_p)
        # 能耗初始化
        for uav in world.UAVs:
            uav.state.energy = Energy
        self.reset_service(world)

    # 奖励函数
    def reward(self, world):
        capacity_list, reward_list = self.get_sum_capacity(world)
        # capacity_sum = np.sum(capacity_list)
        # reward = capacity_sum
        # reward = []
        Jain_index = self.get_Jain_index(world)
        reward = Jain_index * np.array(capacity_list) / 100  # 归一化
        # for i in range(world.num_UAVs):
        #     reward.append(reward_list[i])  # 未归一化奖励
        return capacity_list, reward

    # 计算Jain系数
    def get_Jain_index(self, world):
        volumns = []
        for landmark in world.landmarks.values():
            volumns.append(landmark.sum_throughput)
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
        obs_weight = []
        for landmark in world.landmarks.values():
            # obs_weight.append(landmark.weight)
            obs_weight.append(landmark.sum_throughput)
        obs_weight_norm = []
        for obs in obs_weight:
            obs_weight_norm.append(obs / max(obs_weight))
        # obs_weight_norm.shape:(30,)
        # print('obs_weight_norm.shape:{}'.format(np.array(obs_weight_norm).shape))
        return np.concatenate((np.concatenate(obs_position), np.array(obs_weight_norm)))

    def step(self, world):
        # 标致位，用来判断UAV此次运动是否越界
        is_out_bound = [False for _ in range(world.num_UAVs)]
        # 时隙自增
        world.t += 1
        # reset 服务关联
        for uav in world.UAVs:
            uav.associator.clear()
        for landmark in world.landmarks.values():
            landmark.connected = False
        # 更新位置和能耗
        for i, uav in enumerate(world.UAVs):
            distance_x = uav.action.distance_x * uav.max_distance
            distance_y = uav.action.distance_y * uav.max_distance
            uav.state.pos[0] += distance_x
            uav.state.pos[1] += distance_y

            if uav.state.pos[0] < 0 or uav.state.pos[0] > Range or uav.state.pos[1] < 0 or uav.state.pos[1] > Range:
                is_out_bound[i] = True
                continue

            # uav统计覆盖范围内的地面用户数量
            for landmark in world.landmarks.values():
                if landmark.connected is False and np.sqrt(np.sum((landmark.state.pos - uav.state.pos) ** 2)) <= 60:
                    world.landmarks[str(landmark.id)].connected = True
                    # world.landmarks[str(landmark.id)].sum_throughput += self.get_capacity(uav, landmark)
                    uav.associator.append(landmark.id)

            # 计算uav和覆盖范围内的用户的数据传输率
            for landmark_id in uav.associator:
                world.landmarks[str(landmark_id)].sum_throughput += self.get_capacity(uav,
                                                                                      world.landmarks[str(landmark_id)])

            # 更新landmark的平均数据率和公平比例权重
        
        return is_out_bound

    def get_done(self, world):
        done = []
        for uav in world.UAVs:
            if uav.state.pos[0] < 0 or uav.state.pos[0] > Range or uav.state.pos[1] < 0 or uav.state.pos[1] > Range:
                done.append(True)
            else:
                done.append(False)
        return done

    def reset_service(self, world):
        for uav in world.UAVs:
            uav.state.curServiceNum = 0
            uav.associator = []
        for landmark in world.landmarks.values():
            landmark.connected = False

    # 根据香农计算当前信道容量
    def get_sum_capacity(self, world):
        capacity_list = []
        reward_list = []
        for uav in world.UAVs:
            capacity = 0
            reward = 0
            for id in uav.associator:
                landmark = world.landmarks[str(id)]
                probability_los = self.get_probability(uav.state.pos, landmark.state.pos)  # 获得LoS概率
                # print("建立LoS的概率为{:.4f}".format(probability_los))
                pathLoss = self.get_passLoss(uav.state.pos, landmark.state.pos, probability_los)  # 获得平均路径损失
                capacity += (Bandwidth / len(uav.associator)) * math.log(1 + P_tr * (1 / pathLoss) / sigma_power, 2)
                reward += (Bandwidth / len(uav.associator)) * math.log(1 + P_tr * (1 / pathLoss) / sigma_power,
                                                                       2)   # 根据香农公式计算信道容量
            capacity_list.append(capacity)
            reward_list.append(reward)
        return capacity_list, reward_list

    # 计算某个UAV和地面设备之间的信道容量
    def get_capacity(self, uav, landmark):
        probability_los = self.get_probability(uav.state.pos, landmark.state.pos)  # 获得LoS概率
        pathLoss = self.get_passLoss(uav.state.pos, landmark.state.pos, probability_los)  # 获得平均路径损失
        # 根据香农公式计算信道容量
        capacity = (Bandwidth / len(uav.associator)) * math.log(1 + P_tr * (1 / pathLoss) / sigma_power, 2)
        return capacity


    def get_probability(self, uav_pos, landmark_pos):
        r = np.sqrt(np.sum((landmark_pos - uav_pos) ** 2))
        eta = 0
        if r == 0:
            eta = (180 / math.pi) * math.pi / 2  # 单位是°
        else:
            eta = (180 / math.pi) * np.arctan(H / r)
        # print("eta:{}".format(eta))
        # print(A)
        # print(B)
        probability_los = float(1 / (1 + A * np.exp(-B * (eta - A))))
        # print(probability_los)
        return probability_los

    def get_passLoss(self, uav_pos, landmark_pos, probability_los):
        distance = self.get_distance(uav_pos, landmark_pos)
        pathLoss_LoS = LoS * (4 * math.pi * F * 1e9 * distance / 3e8) ** 2
        pathLoss_NLoS = NLoS * (4 * math.pi * F * 1e9 * distance / 3e8) ** 2
        return probability_los * pathLoss_LoS + (1 - probability_los) * pathLoss_NLoS

    def get_distance(self, uav_pos, landmark_pos):
        distance = np.sqrt(np.sum((landmark_pos - uav_pos) ** 2) + H ** 2)
        return distance

    def greedy_step(self, world):
        # 初始化
        # 标致位，用来判断UAV此次运动是否越界
        is_out_bound = [False for _ in range(world.num_UAVs)]
        # 时隙自增
        world.t += 1
        # reset 服务关联
        for uav in world.UAVs:
            uav.associator.clear()
        for landmark in world.landmarks.values():
            landmark.connected = False

        action_space_x = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        action_space_y = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
        capacity_list = []
        for uav in world.UAVs:
            pos_x = uav.state.pos[0]
            pos_y = uav.state.pos[1]
            capacity = 0
            action_x = 0
            action_y = 0

            for i in action_space_x:
                for j in action_space_y:
                    pos_x = pos_x + i * uav.max_distance
                    pos_y = pos_y + j * uav.max_distance
                    if pos_x < 0 or pos_x > Range or pos_y < 0 or pos_y > Range:
                        pos_x = pos_x - i
                        pos_y = pos_y - j
                        break

                    ## 计算出当前情况下该UAV的用户关联
                    for landmark in world.landmarks.values():
                        uav_pos = np.array([pos_x, pos_y])
                        if landmark.connected is False and np.sqrt(
                                np.sum((landmark.state.pos - uav_pos) ** 2)) <= 60:
                            world.landmarks[str(landmark.id)].connected = True
                            # world.landmarks[str(landmark.id)].sum_throughput += self.get_capacity(uav, landmark)
                            uav.associator.append(landmark.id)

                    ## 计算该UAV产生的吞吐量
                    cur_capacity = 0
                    for id in uav.associator:
                        landmark = world.landmarks[str(id)]
                        probability_los = self.get_probability(uav.state.pos, landmark.state.pos)  # 获得LoS概率
                        pathLoss = self.get_passLoss(uav.state.pos, landmark.state.pos, probability_los)  # 获得平均路径损失
                        cur_capacity += (Bandwidth / len(uav.associator)) * math.log(
                            1 + P_tr * (1 / pathLoss) / sigma_power, 2)

                    if cur_capacity > capacity:
                        capacity = cur_capacity
                        action_x = i
                        action_y = j

                    ## 重置
                    pos_x = pos_x - i * uav.max_distance
                    pos_y = pos_y - j * uav.max_distance
                    for landmark in world.landmarks.values():
                        if landmark.id in uav.associator:
                            landmark.connected = False
                    uav.associator.clear()

            print(action_x, action_y)

            # 更新该UAV的位置和关联用户
            capacity_list.append(capacity)
            uav.state.pos += np.array([action_x*uav.max_distance, action_y*uav.max_distance])
            for landmark in world.landmarks.values():
                uav_pos = np.array([pos_x, pos_y])
                if landmark.connected is False and np.sqrt(
                        np.sum((landmark.state.pos - uav_pos) ** 2)) <= 60:
                    world.landmarks[str(landmark.id)].connected = True
                    # world.landmarks[str(landmark.id)].sum_throughput += self.get_capacity(uav, landmark)
                    uav.associator.append(landmark.id)

        return capacity_list




if __name__ == '__main__':
    sc = Scenario()
    a = np.array([816.21, 531.57])
    b = np.array([752.02, 523.63])
    probability = sc.get_probability(a, b)
    print(probability)
