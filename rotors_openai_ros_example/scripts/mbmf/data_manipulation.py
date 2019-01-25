import numpy as np
import math


np.set_printoptions(precision=10)
    
def from_observation_to_usablestate(states, just_one=True):
    if(just_one):
            curr_item = np.copy(states)
            body_pos = -1*curr_item[3:6]
            # rotate_mat = np.array(curr_item[0:9]).reshape(3,3)
            # body_rpy = euler_angles_from_rotation_matrix(rotate_mat) #9 vals of rot mat --> 6 vals (cos sin of rpy)
            body_rpy = curr_item[0:3]
            body_rpy_sc = [np.cos(body_rpy[0]), np.sin(body_rpy[0]), np.cos(body_rpy[1]), np.sin(body_rpy[1]), np.cos(body_rpy[2]), np.sin(body_rpy[2])]
            body_vel = curr_item[6:9]
            body_rpy_vel = curr_item[9:12]
            full_item = np.concatenate((body_rpy_sc, body_pos, body_vel, body_rpy_vel), axis=0)
            return full_item
    else:
        new_states=[]
        for i in range(len(states)): #for each rollout
            full_item = from_observation_to_usablestate(states[i], True)
            new_states.append(full_item)
        return new_states
    
def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi