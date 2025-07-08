import math
import os, sys
import numpy as np

class IMPEDANCE(object):

    def MassSpringDamper(self, state, t, F):
        x = state[0]
        xd = state[1]
        m = 5
        b = 2
        k = 0.1
        # m = 5
        # b = 2
        # k = 0.1
        xdd = -b/m * xd + (-k/m) * x + F/m
        return [xd, xdd]
    
    def impedance_obs(self, curr_drone_pos, obstacle_center, rr_imp):
        F_coeff = 0.45

        dir_to_center = obstacle_center - curr_drone_pos
        dir_to_center /= np.linalg.norm(dir_to_center)  # Normalize direction vector
        deflection_distance = F_coeff * rr_imp  
        curr_drone_pos -= (deflection_distance * dir_to_center + 0.01)  
        return curr_drone_pos 
    
    def impedance(self, leader_vel, imp_pose_prev, imp_vel_prev, time_step):
        F_coeff = 0.45
        # time_step = time.time() - time_prev
        # time_prev = time.time()
        t = [0., time_step]
        F = 0.01 * F_coeff * leader_vel
        num_drones = len(leader_vel)

        imp_pose = np.zeros_like(imp_pose_prev)
        imp_vel = np.zeros_like(imp_vel_prev)

        for s in range(num_drones):
            state0 = [imp_pose_prev[s], imp_vel_prev[s]]
            state = self.MassSpringDamper(state0, t, F[s])
            imp_pose[s] = state[0]*time_step
            imp_vel[s] = state[1]*time_step
        return imp_pose, imp_vel 
    
    