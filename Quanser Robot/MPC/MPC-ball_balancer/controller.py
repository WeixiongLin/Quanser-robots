import numpy as np
from Hive import Hive
from Hive import Utilities
from canyon import *

class MPC(object):
    def __init__(self, env, config):
        self.env = env
        mpc_config = config["mpc_config"]
        self.horizon = mpc_config["horizon"]
        self.numb_bees = mpc_config["numb_bees"]
        self.max_itrs = mpc_config["max_itrs"]
        self.gamma = mpc_config["gamma"]
        self.action_low = mpc_config["action_low"]
        self.action_high = mpc_config["action_high"]
        self.evaluator = Evaluator(self.gamma)

    def act(self, state, dynamic_model):
        '''
        Optimize the action by Artificial Bee Colony algorithm
        :param state: (numpy array) current state
        :param dynamic_model: system dynamic model
        :return: (float) optimal action
        '''
        self.evaluator.update(state, dynamic_model)
        optimizer = Hive.BeeHive( lower = [float(self.action_low)] * self.horizon,
                                  upper = [float(self.action_high)] * self.horizon,
                                  fun = self.evaluator.evaluate,
                                  numb_bees = self.numb_bees,
                                  max_itrs = self.max_itrs,
                                  verbose=False)
        # optimizer = Canyon( lower =   [float(self.action_low)] * self.horizon, upper = [float(self.action_high)] * self.horizon,
        #                                   fun = self.evaluator.evaluate)
        cost = optimizer.run()
        #print("Solution: ",optimizer.solution[0])
        #print("Fitness Value ABC: {0}".format(optimizer.best))
        # Uncomment this if you want to see the performance of the optimizer
        #Utilities.ConvergencePlot(cost)
        return optimizer.solution[0]

class Evaluator(object):
    def __init__(self, gamma=0.8):
        self.gamma = gamma
        self.Q = np.diag([1e-2, 1e-2, 1e-0, 1e-0, 1e-4, 1e-4, 1e-2, 1e-2])  # see dim of state space
        self.R = np.diag([1e-4, 1e-4])  # see dim of action space
        self.min_rew = 1e-4
        
        
    def update(self, state, dynamic_model):
        self.state = state
        self.dynamic_model = dynamic_model

    def evaluate(self, actions):
        actions = np.array(actions)
        horizon = actions.shape[0]
        rewards = 0
        state_tmp = self.state.copy()
        for j in range(horizon):
            ##changing here
            input_data = np.concatenate( (state_tmp,[actions[j][0],actions[j][1]]) )
        #    print(input_data)
            ###end changing
            state_dt = self.dynamic_model.predict(input_data)
            state_tmp = state_tmp + state_dt[0]
            rewards -= (self.gamma ** j) * self.get_reward(state_tmp, actions[j])
        return rewards

    def get_reward(self,obs, action):

        err_s = (self.state - obs).reshape(-1,)  # or self._state
        err_a = action.reshape(-1,)
        quadr_cost = err_s.dot(self.Q.dot(err_s)) + err_a.dot(self.R.dot(err_a))
        state_max = np.array([np.pi/4., np.pi/4., 0.15, 0.15, 4.*np.pi, 4.*np.pi, 0.5, 0.5])
        obs_max = state_max.reshape(-1, )
        actmax = np.array([5.0, 5.0])
        act_max = actmax.reshape(-1, )

        max_cost = obs_max.dot(self.Q.dot(obs_max)) + act_max.dot(self.R.dot(act_max))
        # Compute a scaling factor that sets the current state and action in relation to the worst case
        self.c_max = -1.0 * np.log(self.min_rew) / max_cost


        # Calculate the scaled exponential
        rew = np.exp(-self.c_max * quadr_cost)  # c_max > 0, quard_cost >= 0
        return float(rew)
