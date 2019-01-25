import numpy as np
from cost_functions import trajectory_cost_fn
import time
import math
import numpy as np

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        return self.env.action_space.sample()
	#pass

class RefMPCController(Controller):
    def __init__(self, env, call_mpc_fn):
        self.env = env
        self.mpc_fn = call_mpc_fn

    def get_action(self, state):
        return self.mpc_fn(state)


class MPCcontroller(Controller):
    def __init__(self, env, dyn_model, horizon=5, cost_fn=None, num_simulated_paths=10):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        # print(state.shape[0])
        # horizon * num_paths* state_dim or action_dim 

        trajs = np.array([self.env.action_space.sample() for p in range(self.horizon*self.num_simulated_paths)]).reshape((self.horizon, self.num_simulated_paths, -1))
        #print "Trajs dimension", trajs.shape
        #TODO: action_dim #n_paths*horizon*action_dim
        accum_states = np.zeros((self.horizon, self.num_simulated_paths, state.shape[0]))
        accum_next_states = np.zeros((self.horizon, self.num_simulated_paths, state.shape[0]))
        states = np.ones((self.num_simulated_paths,1))*state # n_paths*state_dim
        #print "state", state
        #print "accum_states", accum_states
        #print "accum_states shape", accum_states.shape
        for time_idx in range(self.horizon):
            actions = trajs[time_idx, :, :] # n_paths * action_dim
            #print "action dimension", actions.shape
            next_states = self.dyn_model.predict(states, actions)
            #print "next states dimension", next_states.shape
            accum_states[time_idx,:, :] = states
            accum_next_states[time_idx,:, :] = next_states
            states = next_states.copy() #set the states to be next_states for next time_idx
        cost = trajectory_cost_fn(self.cost_fn, accum_states, trajs, accum_next_states)
            #print "cost shape", cost.shape
            #print "min cost", np.min(cost)
            #print "min cost arg way", cost[np.argmin(cost)]
        action = trajs[0,np.argmin(cost),:] # first action for min rollout
        return action.flatten() 

k_yaw = 1.8
def yaw_controller(ref_yaw, current_yaw, ref_yaw_rate=0):
    yaw_error = ref_yaw - current_yaw
    if abs(yaw_error) > math.pi:
        yaw_error = yaw_error - 2.0*math.pi
    else:
        yaw_error = yaw_error + 2.0*math.pi
    yaw_rate_cmd = k_yaw * yaw_error + ref_yaw_rate
    return np.clip(yaw_rate_cmd, -yaw_rate_limit, yaw_rate_limit)
