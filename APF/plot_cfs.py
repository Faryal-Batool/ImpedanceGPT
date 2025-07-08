import matplotlib.pyplot as plt
import numpy as np
import re

def extract_positions(filename):
    times = []
    cf2_x, cf2_y = [], []
    cf3_x, cf3_y = [], []
    
    with open(filename, 'r') as file:
        for line in file:
            time_match = re.search(r'Time: ([\d.]+)', line)
            if time_match:
                times.append(float(time_match.group(1)))

            match = re.search(r'CF2: \[([-?\d.]+), ([-?\d.]+),', line)
            if match:
                cf2_x.append(float(match.group(1)))
                cf2_y.append(float(match.group(2)))
            
            match = re.search(r'CF3: \[([-?\d.]+), ([-?\d.]+),', line)
            if match:
                cf3_x.append(float(match.group(1)))
                cf3_y.append(float(match.group(2)))
    
    return times, (cf2_x, cf2_y), (cf3_x, cf3_y)

import re

def extract_goal_and_obstacles(filename):
    goal_x, goal_y = [], []
    obstacles_x, obstacles_y = [], []
    
    with open(filename, 'r') as file:
        in_obstacles_block = False
        obstacles_str = ""
        
        for line in file:
            # Extract goal position
            goal_match = re.search(r'Goal: \[([-?\d.]+), ([-?\d.]+),', line)
            if goal_match:
                goal_x.append(float(goal_match.group(1)))
                goal_y.append(float(goal_match.group(2)))
            
            # Start of obstacles block
            if "Obstacles:" in line:
                in_obstacles_block = True
                obstacles_str = ""
            
            # Capture obstacle coordinates
            if in_obstacles_block:
                obstacles_str += line.strip()  # Accumulate the obstacle lines
                # If we encounter the closing bracket of obstacles
                if ']' in line and 'Obstacles:' not in line:
                    in_obstacles_block = False
                    # Now process the obstacles string
                    obstacle_positions = re.findall(r'\[([-?\d.]+)\s+([-?\d.]+)', obstacles_str)
                    for pos in obstacle_positions:
                        obstacles_x.append(float(pos[0]))
                        obstacles_y.append(float(pos[1]))
    
    return (goal_x, goal_y), (obstacles_x, obstacles_y)



def plot_trajectories(cf2, cf3, goal, obstacles):
    plt.figure(figsize=(10, 8))

    # Plot CF2 and CF3 trajectories
    # plt.plot(cf2[0], cf2[1], marker='o', label='CF2', linestyle='-', color='blue')
    # plt.plot(cf3[0], cf3[1], marker='s', label='CF3', linestyle='-', color='green')

    # Plot goal position
    plt.scatter(goal[0], goal[1], c='red', marker='*', s=200, label='Goal')

    # Plot obstacle positions
    plt.scatter(obstacles[0], obstacles[1], c='black', marker='x', label='Obstacles')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('CF2 and CF3 Trajectories with Goal and Obstacles')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
filename_trajectories = 'vicon_data/exp1_2_drone_static/drone_poses_exp1.txt'  # Change this to the actual filename for drone trajectories
filename_goal_obstacles = 'vicon_data/exp1_2_drone_static/goal_and_obstacles_exp1.txt'  # Change this to the actual filename for goal and obstacles

# Extract drone trajectories
times, cf2, cf3 = extract_positions(filename_trajectories)

# Extract goal and obstacles positions
goal, obstacles = extract_goal_and_obstacles(filename_goal_obstacles)

# Plot everything
plot_trajectories(cf2, cf3, goal, obstacles)
