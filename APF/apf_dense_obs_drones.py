import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from apf import APF
from impedance import IMPEDANCE
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.CtrlAviary2 import CtrlAviary
import csv

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 5
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 18
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_NUM_CARS = 0 

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        NUM_CARS=DEFAULT_NUM_CARS,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    
    ############################## Initial positions, targets, and orientations Multiple Cases ##############################

    # CASE III
    height = 1.5
    center_position = [-2, 1.5, height]  # Base position of the center drone
    separation_distance = 0.5 # Distance between drones

    # Initialize positions (using angle-based separation)
    INIT_XYZS = np.zeros((num_drones, 3))

    rotation = 90
    INIT_XYZS[0] = center_position
    for j in range(1, num_drones):
        angle = (2*np.pi*(j-1))/(num_drones-1) + rotation / 57.3
        # For follower drones, calculate position based on angle and separation distance
        x_offset = separation_distance * np.cos(angle)  # x = center_x + separation_distance * cos(angle)
        y_offset = separation_distance * np.sin(angle)  # y = center_y + separation_distance * sin(angle)
        INIT_XYZS[j] = [center_position[0] + x_offset, center_position[1] + y_offset, center_position[2]]

    INIT_RPYS = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]])

        
    OBST_POS = np.array([  [0.3, 0.5, 0, 0.20],
                           [-0.3, -0.54, 0, 0.20],
                           [-1, 1.2, 0, 0.20],
                           [-1, 0.6, 0, 0.20],
                           [1.3, 0, 0, 0.20],
                           [1, -1.7, 0, 0.20]])


    ##### NOT USED ###
    GROUND_OBS = np.array([[-1.5,-0.36,0]])
    target_pos=np.array([[2, -1.5, 1.5]])
    dx = target_pos[0][0] - INIT_XYZS[0][0] 
    dy = target_pos[0][1] - INIT_XYZS[0][1] 
    theta_goal = np.arctan2(dy, dx)
    INIT_XYZS_C = np.array([[INIT_XYZS[0][0], INIT_XYZS[0][1], 0]]) 
    INIT_RPYS_C = np.array([[0, 0, theta_goal] for _ in range(NUM_CARS)])
    ##### NOT USED ###
    obs_def = 0.65 # rr_imp
    obs_radius = 0.08 + obs_def # obs radius + rr_imp
    
    ##################################Initializing parameters for plotting###############################
    current_positions_drone = np.empty((0, 3))
    current_positions_drone = [np.empty((0, 3)) for _ in range(num_drones)]  # Initialize an empty np.array for each drone

    current_positions_side_drone = np.empty((0, 3))
    leader_poses =  np.empty((0, 3))

    # side_drone_poses = []
    side_drone_poses = [np.empty((0, 2)) for _ in range(num_drones)]  # Initialize an empty np.array for each drone

    #################################### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        obst_pos=OBST_POS,  
                        target_pos=target_pos,
                        ground_obs=GROUND_OBS,
                        physics=physics,
                        neighbourhood_radius=10,    
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui,
                        NUM_CARS=NUM_CARS,  
                        INIT_XYZS_C=INIT_XYZS_C,  
                        INIT_RPYS_C=INIT_RPYS_C
                        )
    


    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    ######################################### Initialize the controllers ############################
    if drone in [DroneModel.RACE, DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    ############################################ Run the simulation ####################################

    action = np.zeros((num_drones,4))
    START = time.time()
    control_input = np.zeros((num_drones,3))
    imp_pose_prev = np.zeros(3); imp_vel_prev = np.zeros(3); imp_time_prev = 0

    apf = APF(start=env.INIT_XYZS[0], goal=env.target_pos[0])
    impedance = IMPEDANCE()

    file = open('/home/dzmitry/IndiaGrant/rl_environment/gym_pybullet_drones/examples/drones.csv', mode='w', newline='')
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['time', 'd_x', 'd_y', 'r_x', 'r_y'])
    start_time = time.time()

    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        current_time = time.time() - start_time

        obs, reward, terminated, truncated, info = env.step(action)

        ######################## Leader Drone ###########################

        cur_pos = obs[0,0:3]
        attr_force = apf.getAttractiveForce(cur_pos=cur_pos, tgt_pos=np.array(env.target_pos[0]))
        rep_force = apf.getRepulsiveForce(cur_pos=cur_pos[0:2],obstacles=env.obst_pos[:,0:2])
        rep_force = np.hstack([rep_force,[0]])
        net_force = (apf.zeta * attr_force + apf.eta * rep_force) * 0.08  # Add damping factor
        # Add velocity limiting
        norm = np.linalg.norm(net_force)
        if norm > apf.max_v: # Add velocity limiting
            net_force = (net_force / norm) * apf.max_v
        control_input[0, 0:3] = net_force

        ##### Compute control for the current way point for leader drone #############
        for j in range(1):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=obs[j,0:3] + control_input[j],   
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )
        x, y, z = obs[0,0], obs[0,1], obs[0,2]    # leader x,y,z
        vx, vy, vz = obs[0,6], obs[0,7], obs[0,8]  # leader vx, vy, vz

        ######################## Impedance-based Drones ###########################
        
        leader_pose = [x,y,z]

        #########################storing leader poses for plotting#########################
        leader_poses = np.vstack([leader_poses, leader_pose])

        leader_vel = np.array([vx, vy, vz])

        for j in range(1, num_drones):
            time_step = time.time() - imp_time_prev - start_time
            imp_pose, imp_vel = impedance.impedance(leader_vel, imp_pose_prev, imp_vel_prev, time_step)
            angle = (2*np.pi*(j-1))/(num_drones-1) + rotation / 57.3
            x_offset = separation_distance * np.cos(angle)  # x = center_x + separation_distance * cos(angle)
            y_offset = separation_distance * np.sin(angle)  # y = center_y + separation_distance * sin(angle)
            drones_pose = leader_pose + (0.2 * imp_pose) + np.array([x_offset, 
                                                                     y_offset, 
                                                                     0])

            
            for obstacle in env.obst_pos:
             if np.linalg.norm(drones_pose[:2] - obstacle[:2]) < (obs_radius):
                        drones_pose[:2] = impedance.impedance_obs(drones_pose[:2], obstacle[0:2], obs_def)
            target = [drones_pose[0], drones_pose[1], height]
            imp_pose_prev = imp_pose
            imp_vel_prev = imp_vel
            imp_time_prev = current_time

            # side_drone_poses = np.vstack([side_drone_poses, drones_pose[0:2]])
            # Append the drone's position to its respective array

            side_drone_poses[j] = np.vstack([side_drone_poses[j], drones_pose[0:2]])  # Storing x and y

            state_drone = env._getDroneStateVector(j)  # Get the state for each drone (0 for the first drone, etc.)
            current_position_drone = state_drone[0:3]  # Extract [x, y, z] position

            # Append current position to the respective drone's trajectory
            current_positions_drone[j] = np.vstack([current_positions_drone[j], current_position_drone])

            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                        state=obs[j],
                                                        target_pos=target,   
                                                        target_rpy=INIT_RPYS[j, :]
                                                        )
        camera_distance = 5  # Adjust the zoom (distance of the camera from the target)
        camera_yaw = -90      # Camera angle along the z-axis (left-right)
        camera_pitch = -89   # Camera angle along the y-axis (up-down)
        camera_target = [-0.47, 0, 0]  # Center the camera on the drone
        # # Change camera view
        # p.resetDebugVisualizerCamera(
        #     cameraDistance=camera_distance,
        #     cameraYaw=camera_yaw,
        #     cameraPitch =camera_pitch,
        #     cameraTargetPosition=camera_target
        # )
        # # env.render()

        
                                                                                                            
        # ### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    if plot:
        plt.figure(figsize=(5, 5))
        ################### obstacles positons##############################################
        obstacles_pos = np.array(OBST_POS)
        for i, obs in enumerate(obstacles_pos):
            if i == 0:  # Add the label only once for the first circle
                plt.gca().add_artist(plt.Circle((obs[0], obs[1]), apf.d_0-0.2, color='lightgray', fill=True, label='Obstacle deflection region'))
                plt.gca().add_artist(plt.Circle((obs[0], obs[1]), 0.1, color='black', fill=True, label='Obstacles'))
            else:
                plt.gca().add_artist(plt.Circle((obs[0], obs[1]), apf.d_0-0.2, color='lightgray', fill=True))
                plt.gca().add_artist(plt.Circle((obs[0], obs[1]), 0.1, color='black', fill=True))

        ##############################Initial and Target Position ##########################33
        plt.scatter(INIT_XYZS[0, 0], INIT_XYZS[0, 1], color='#DC143C', label='Start Position', s=190, marker='o', edgecolor='black', linewidth=2)  # Increase size (adjust 's' for bigger/smaller markers)
        plt.scatter(target_pos[0, 0], target_pos[0, 1], color='#00FF00', label='Target Position', s=190, marker='o', edgecolor='black', linewidth=2)  # Increase size

        #######################Drone actual and apf path ####################################        
        x_leader_apf = leader_poses[:,0]
        y_leader_apf = leader_poses[:,1]
        plt.plot(x_leader_apf, y_leader_apf, linestyle='--', color='k', label= 'Planned APF Path (Drone)')

        # Create or open the text file to save the drone positions
        with open('/home/dzmitry/IndiaGrant/rl_environment/gym_pybullet_drones/examples/dataset/4_drones/scenario10/exp1/drone_positions.txt', 'w') as file:
            # Loop through each drone and save its positions
            for j in range(num_drones):
                # Get the positions of the current drone
                current_position_drone = current_positions_drone[j]
                
                # Write the label for the drone
                file.write(f"Drone {j+1} Positions:\n")
                
                # Save each position (x, y, z) for the current drone
                np.savetxt(file, current_position_drone, header="x, y, z", fmt='%0.4f')
                
                # Add a newline for better separation between drones' positions
                file.write("\n\n")

        # ############################# Side droens actual and apf path #####################################
        for i in range((num_drones)):
            side_drone_poses_x = side_drone_poses[i][:, 0]
            side_drone_poses_y = side_drone_poses[i][:, 1]
            x_trajectory = current_positions_drone[i][:, 0]  # Extract x positions
            y_trajectory = current_positions_drone[i][:, 1]  # Extract y positions
            plt.plot(x_trajectory, y_trajectory, label=f"Drone {j+1}")  # Plot and label each drone
            plt.plot(side_drone_poses_x, side_drone_poses_y, linestyle='--', color='#DAA520', label= 'Planned Impedance Path (Ground Robot)')            

        plt.axis('equal')
        plt.xlabel("X Position (meters)")
        plt.ylabel("Y Position (meters)")
        # plt.legend()  # Show legend to differentiate drones
        plt.grid(True)
        plt.savefig('/home/dzmitry/IndiaGrant/rl_environment/gym_pybullet_drones/examples/dataset/4_drones/scenario10/exp1/drone_positions.png')

        plt.show()

    #### Close the environment #################################
    file.close()
    env.close()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))



