import numpy as np
import random

"""
canyon 是自己写的 optimizer, 可以无缝替换到 controller.py 中 Hive.BeeHive 的位置
思路就是模仿加农炮打靶, 根据 Reward 不断优化自己的 action
"""


class Canyon(object):
    def __init__(self, lower, upper, fun):
        self.lower = lower
        self.upper = upper
        self.horizen = len(self.lower)
        self.evaluate = fun
        self.solution = None
    
    def run(self):
        epsilon = 0.1  # 只要 action 的变化量大于 epsilon 就说明没有收敛
        action_delta = 1  # action 更新的变化量
        reward_episode = 0
        render = False
        left_action = self.lower
        right_action = self.upper

        cur_action = []
        for tmp in range(self.horizen):
            action_0 = left_action[tmp][0] + random.random() * (right_action[tmp][0] - left_action[tmp][0])
            action_1 = left_action[tmp][1] + random.random() * (right_action[tmp][1] - left_action[tmp][1]) 
            cur_action.append( [action_0, action_1] )
        cur_action = np.array(cur_action)

        label_tmp = []

        while(action_delta > epsilon):
            action_delta = 0
            for step in range(self.horizen):  # 依次更新每一个 action
                left_reward = self.evaluate(left_action)
                right_reward = self.evaluate(right_action)
                if left_reward > right_reward:
                    action_delta += ((cur_action[step][0] - right_action[step][0])**2+(cur_action[step][1] - right_action[step][1])**2)**0.5
                    left_action[step] = cur_action[step]
                    cur_action[step] = (right_action[step] + cur_action[step]) / 2
                else:
                    action_delta += ((cur_action[step][0] - right_action[step][0])**2+(cur_action[step][1] - right_action[step][1])**2)**0.5
                    right_action[step] = cur_action[step]
                    cur_action[step] = (left_action[step] + cur_action[step]) / 2
        self.solution = cur_action
        return cur_action