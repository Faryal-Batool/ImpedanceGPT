import math
import os, sys
import numpy as np
from scipy.spatial.distance import cdist

class APF(object):
    '''
    Class for Artificial Potential Field(APF) motion planning.

    Parameters
    ----------
    start: tuple
        start point coordinate
    goal: tuple
        goal point coordinate
    obstacles: np.ndarray
        obstacles coordinate
    '''
    def __init__(self, start: tuple, goal: tuple):
        self.zeta = 1.5
        self.eta = 1.5
        self.d_0 = 0.45
        self.max_v = 1.5
   
    def getRepulsiveForce(self, cur_pos: np.ndarray,obstacles: np.ndarray):
        '''
        Get the repulsive  force of APF.

        Return
        ----------
        rep_force: np.ndarray
            the repulsive force of APF
        '''
        # print(f'cur_pos = {cur_pos} and obstacles = {obstacles}')
        # print(f'shape of obstacles = {np.shape(self.obstacles)} and shape of cur_pos = {np.shape(cur_pos)}')
        # D = cdist(self.obstacles[np.newaxis, :], cur_pos[np.newaxis, :])

        D = cdist(obstacles,cur_pos[np.newaxis, :])
        # print(f'D = {D}')   
        rep_force = (1 / D - 1 / self.d_0) * (1 / D) ** 2 * (cur_pos - obstacles)
        valid_mask = np.argwhere((1 / D - 1 / self.d_0) > 0)[:, 0]
        rep_force = np.sum(rep_force[valid_mask, :], axis=0)
        # print(f'rep_force = {rep_force}')
        if not np.all(rep_force == 0):
            rep_force = rep_force / np.linalg.norm(rep_force)
        
        
        return rep_force
    
    def getAttractiveForce(self, cur_pos: np.ndarray, tgt_pos: np.ndarray):
        '''
        Get the attractive force of APF.

        Parameters
        ----------
        cur_pos: np.ndarray
            current position of robot
        tgt_pos: np.ndarray
            target position of robot

        Return
        ----------
        attr_force: np.ndarray
            the attractive force
        '''
        attr_force = tgt_pos - cur_pos
        if not np.all(attr_force == 0):
            attr_force = attr_force / np.linalg.norm(attr_force)
        
        return attr_force
