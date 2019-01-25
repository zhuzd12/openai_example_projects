import numpy as np
import math

#========================================================
# 
# Environment-specific cost functions:
#

def cheetah_cost_fn(state, action, next_state):
    if len(state.shape) > 1:

        heading_penalty_factor=10
        scores=np.zeros((state.shape[0],))

        #dont move front shin back so far that you tilt forward
        front_leg = state[:,5]
        my_range = 0.2
        scores[front_leg>=my_range] += heading_penalty_factor

        front_shin = state[:,6]
        my_range = 0
        scores[front_shin>=my_range] += heading_penalty_factor

        front_foot = state[:,7]
        my_range = 0
        scores[front_foot>=my_range] += heading_penalty_factor

        scores-= (next_state[:,17] - state[:,17]) / 0.01 #+ 0.1 * (np.sum(action**2, axis=1))
        return scores

    heading_penalty_factor=10
    score = 0

    #dont move front shin back so far that you tilt forward
    front_leg = state[5]
    my_range = 0.2
    if front_leg>=my_range:
        score += heading_penalty_factor

    front_shin = state[6]
    my_range = 0
    if front_shin>=my_range:
        score += heading_penalty_factor

    front_foot = state[7]
    my_range = 0
    if front_foot>=my_range:
        score += heading_penalty_factor

    score -= (next_state[17] - state[17]) / 0.01 #+ 0.1 * (np.sum(action**2))
    return score

kGravity = 9.81
pelican_mass = 1.0
weight_vector = np.array([80, 80, 120, 80, 80, 100, 10, 10, 50, 50, 1])
def quadrotor_control_cost_fn(states, actions, next_states):
    costs = []
    # print(actions.shape)
    # print(states.shape)
    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        lstate = list(state)
        laction = list(action)
        mpc_state = lstate[6:9] + lstate[9:12] + [math.acos(np.clip(state[0],-1.0,1.0)), math.acos(np.clip(state[2],-1.0,1.0))]
        mpc_state = mpc_state + laction[1:3] 
        mpc_state = mpc_state + [state[0]*state[2]*laction[3]-kGravity*pelican_mass]
        state_cost = sum(weight_vector * np.power(np.array(mpc_state), 2))
        costs.append(state_cost)
    return np.array(costs)

#========================================================
# 
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
    trajectory_cost = 0
    # print(actions.shape)
    for i in range(len(actions)):
        trajectory_cost += cost_fn(states[i], actions[i], next_states[i])
    return trajectory_cost
