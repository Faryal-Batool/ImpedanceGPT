#!/usr/bin/env python

import rclpy    
from rclpy.node import Node
import os
import numpy as np
import time
from crazyflie_interfaces.msg import FullState,Position,Hover
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from .apf import APF
from .impedance import IMPEDANCE
import time
from crazyflie_py.crazyflie import CrazyflieServer, TimeHelper

class APF_IMP(Node):
    def __init__(self):

        super().__init__('APF_IMP')
        self.allcfs = CrazyflieServer()
        self.timeHelper = TimeHelper(self.allcfs)

        self.leader = None
        self.cf2 = None
        self.cf3 = None
        self.cf4 = None
        self.previous_leader = np.array([0,0,0])
        self.goal = None
        self.obstacles = None
        self.apf = None
        self.impedance = None
        self.start_time = 0
        self.previous_time = 0
        
        # Initialize file paths for saving poses and other data
        self.folder_path = '/home/dzmitry/ros2_ws/src/swarm_vlm/swarm_vlm/vicon_data'  # Specify the folder path
        os.makedirs(self.folder_path, exist_ok=True)

        # Files to save the data
        self.drone_poses_file = os.path.join(self.folder_path, 'drone_poses_exp1.txt')
        self.goal_and_obstacles_file = os.path.join(self.folder_path, 'goal_and_obstacles_exp1.txt')

        self.check_takeoff = rclpy.task.Future()
        self.check_target = rclpy.task.Future()
        self.check_land = rclpy.task.Future()

        self.num_drones = 3
        self.height = 1.0
        self.center_position = np.array([-1.9, 1.5, self.height])  # Base position of the center drone
        self.separation_distance = 0.55 # Distance between drones
        self.INIT_XYZS = np.zeros((self.num_drones, 3))
        self.rotation = 90
        self.INIT_XYZS[0] = self.center_position
        for j in range(1, self.num_drones):
            angle = (2*np.pi*(j-1))/(self.num_drones-1) + self.rotation / 57.3
            # For follower drones, calculate position based on angle and separation distance
            x_offset = self.separation_distance * np.cos(angle) 
            y_offset = self.separation_distance * np.sin(angle)  
            self.INIT_XYZS[j] = [self.center_position[0] + x_offset, self.center_position[1] + y_offset, self.center_position[2]]
        # print(self.INIT_XYZS)

        self.obs_def = 0.65 # rr_imp
        self.obs_radius = 0.08 + self.obs_def # obs radius + rr_imp
        self.imp_pose_prev = np.zeros(3); 
        self.imp_vel_prev = np.zeros(3); 
        self.imp_time_prev = 0

        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                     history=QoSHistoryPolicy.KEEP_LAST,
                                     depth=1,
                                     deadline=Duration(seconds=0, nanoseconds=1e9/100.0))
        
        # Subscribers
        self.subscription = self.create_subscription(NamedPoseArray, '/poses', self.pose_callback, qos_profile=qos_profile)
    
    def write_to_files(self):
        # Write drone positions (with timestamp) to the drone_poses file
        if self.leader is not None and self.cf2 is not None and self.cf3: # is not None and self.cf4 is not None:
            with open(self.drone_poses_file, 'a') as file:
                file.write(f"Time: {self.current_time2}, Leader: {self.leader}, CF2: {self.cf2}, CF3: {self.cf3}\n") #, CF4: {self.cf4}\n")
        
        # Write goal and obstacle positions to the goal_and_obstacles file
        if self.goal is not None or self.obstacles is not None:
            with open(self.goal_and_obstacles_file, 'a') as file:
                file.write(f"Time: {self.current_time2}\n")
                if self.goal is not None:
                    file.write(f"Goal: {self.goal}\n")
                if self.obstacles is not None:
                    file.write(f"Obstacles: {self.obstacles}\n")
                file.write("\n")

    def pose_callback(self, msg):
        cf1_pose = next((pose for pose in msg.poses if pose.name == 'cf1'), None)
        cf2_pose = next((pose for pose in msg.poses if pose.name == 'cf2'), None)
        cf3_pose = next((pose for pose in msg.poses if pose.name == 'cf3'), None)
        # cf4_pose = next((pose for pose in msg.poses if pose.name == 'cf4'), None)
        
        goal = next((pose for pose in msg.poses if pose.name == 'goal'), None)
        obstacle_names = ['obs_1', 'obs_2', 'gate_1', 'gate_2'] #'obs_4',

        obstacles = []
        for obs_name in obstacle_names:
            obstacle = next((pose for pose in msg.poses if pose.name == obs_name), None)
            if obstacle is not None:
                obstacles.append([obstacle.pose.position.x, obstacle.pose.position.y, obstacle.pose.position.z])
        
        # Store obstacles in the self.obstacles attribute
        self.obstacles = np.array(obstacles)

        if cf1_pose is not None:
            position_msg = Position()
            position_msg.x = cf1_pose.pose.position.x
            position_msg.y = cf1_pose.pose.position.y
            position_msg.z = cf1_pose.pose.position.z
            self.leader = [position_msg.x, position_msg.y, position_msg.z]
        
        if cf2_pose is not None:
            position_msg = Position()
            position_msg.x = cf2_pose.pose.position.x
            position_msg.y = cf2_pose.pose.position.y
            position_msg.z = cf2_pose.pose.position.z
            self.cf2 = [position_msg.x, position_msg.y, position_msg.z]
        
        if cf3_pose is not None:
            position_msg = Position()
            position_msg.x = cf3_pose.pose.position.x
            position_msg.y = cf3_pose.pose.position.y
            position_msg.z = cf3_pose.pose.position.z
            self.cf3 = [position_msg.x, position_msg.y, position_msg.z]

        # if cf4_pose is not None:
        #     position_msg = Position()
        #     position_msg.x = cf4_pose.pose.position.x
        #     position_msg.y = cf4_pose.pose.position.y
        #     position_msg.z = cf4_pose.pose.position.z
        #     self.cf4 = [position_msg.x, position_msg.y, position_msg.z]
        
        if goal is not None:
            position_msg = Position()
            position_msg.x = goal.pose.position.x
            position_msg.y = goal.pose.position.y
            position_msg.z = goal.pose.position.z
            self.goal = [position_msg.x, position_msg.y, position_msg.z]
        # Record the time and write to the files
        self.current_time2 = time.time() - self.start_time
        self.write_to_files()


    def takeoff(self):
        Z = 1.0
        for j in range(self.num_drones):
            self.allcfs.crazyflies[j].takeoff(targetHeight=Z, duration=1.0+Z)
        # self.timeHelper.sleep(5.0)
        if np.linalg.norm(np.array(self.leader)-self.center_position) < 0.25:
            self.check_takeoff.set_result(True)
        
    def land(self):
        Z = 1.0
        self.allcfs.land(targetHeight=0.02, duration=1.0+Z)
        # self.timeHelper.sleep(2.0)
        self.check_land.set_result(True)
    
    def init_apf_imp(self):
        self.apf = APF(start=self.leader, goal=self.goal)
        self.impedance = IMPEDANCE()
        self.start_time = time.time()

    def run(self):
        cur_pos = self.leader
        # print(cur_pos)
        # print(self.goal)
        # print(self.obstacles)
        self.current_time = time.time() - self.start_time
        attr_force = self.apf.getAttractiveForce(cur_pos=cur_pos, tgt_pos=np.array(self.goal))
        rep_force = self.apf.getRepulsiveForce(cur_pos=np.array(cur_pos[0:2]),obstacles=self.obstacles[:,0:2])
        rep_force = np.hstack([rep_force,[0]])
        net_force = (self.apf.zeta * attr_force + self.apf.eta * rep_force) * 0.2  
        norm = np.linalg.norm(net_force)
        if norm > self.apf.max_v: # Add velocity limiting
            net_force = (net_force / norm) * self.apf.max_v
        control_input = net_force
        target_leader = np.array([self.leader[0]+control_input[0],
                                 self.leader[1]+control_input[1],
                                 self.height])
        current_leader_cf = self.allcfs.crazyflies[0]
        current_leader_cf.goTo(target_leader, 0, 0.2)
        
        time_step = self.current_time - self.previous_time 
        leader_vel = (np.array(self.leader) - self.previous_leader) / time_step
        for j in range(1, self.num_drones):
            imp_pose, imp_vel = self.impedance.impedance(leader_vel, self.imp_pose_prev, self.imp_vel_prev, time_step)
            angle = (2*np.pi*(j-1))/(self.num_drones-1) + self.rotation / 57.3
            x_offset = self.separation_distance * np.cos(angle)  
            y_offset = self.separation_distance * np.sin(angle) 
            drones_pose = np.array(self.leader) + (0.2 * imp_pose) + np.array([x_offset, 
                                                                     y_offset, 
                                                                     0])

            for obstacle in self.obstacles:
                if np.linalg.norm(drones_pose[:2] - obstacle[:2]) < (self.obs_radius):
                    drones_pose[:2] = self.impedance.impedance_obs(drones_pose[:2], obstacle[0:2], self.obs_def)
            target = np.array([drones_pose[0], drones_pose[1], self.height])
            current_imp_cf = self.allcfs.crazyflies[j]
            current_imp_cf.goTo(target, 0, 0.05)   

            self.imp_pose_prev = imp_pose
            self.imp_vel_prev = imp_vel
            self.previous_leader = np.array(self.leader)
            self.previous_time = self.current_time

        if np.linalg.norm(np.array(self.leader[0:2])-np.array(self.goal[0:2])) < 0.1:
            self.check_target.set_result(True)
        # Record the time and write to the files
        
        

def main(args=None):
    rclpy.init(args=args)

    # Create the APF_IMP node
    apf_imp = APF_IMP()
    rate = 24

    # Takeoff
    apf_imp.timer = apf_imp.create_timer(1/rate, apf_imp.takeoff)
    rclpy.spin_until_future_complete(apf_imp, apf_imp.check_takeoff)

    # Init APF-IMP
    apf_imp.init_apf_imp()

    # APF-IMP
    apf_imp.timer = apf_imp.create_timer(1/rate, apf_imp.run)
    rclpy.spin_until_future_complete(apf_imp, apf_imp.check_target)

    # Land
    apf_imp.timer = apf_imp.create_timer(1/rate, apf_imp.land)
    rclpy.spin_until_future_complete(apf_imp, apf_imp.check_land)

    # Clean up after the node shuts down
    apf_imp.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()