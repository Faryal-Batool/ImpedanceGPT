import matplotlib.pyplot as plt
import numpy as np
import re

def extract_positions(filename):
    times = []
    leader_x, leader_y = [], []
    cf2_x, cf2_y = [], []
    cf3_x, cf3_y = [], []
    cf4_x, cf4_y = [], []
    
    with open(filename, 'r') as file:
        for line in file:
            time_match = re.search(r'Time: ([\d.]+)', line)
            if time_match:
                times.append(float(time_match.group(1)))

            # Extract position of the leader drone
            match = re.search(r'Leader: \[([-?\d.]+), ([-?\d.]+),', line)
            if match:
                leader_x.append(float(match.group(1)))
                leader_y.append(float(match.group(2)))

            # Extract position of CF2 drone
            match = re.search(r'CF2: \[([-?\d.]+), ([-?\d.]+),', line)
            if match:
                cf2_x.append(float(match.group(1)))
                cf2_y.append(float(match.group(2)))
            
            # Extract position of CF3 drone
            match = re.search(r'CF3: \[([-?\d.]+), ([-?\d.]+),', line)
            if match:
                cf3_x.append(float(match.group(1)))
                cf3_y.append(float(match.group(2)))

            # Extract position of CF4 drone
            match = re.search(r'CF4: \[([-?\d.]+), ([-?\d.]+),', line)
            if match:
                cf4_x.append(float(match.group(1)))
                cf4_y.append(float(match.group(2)))
    
    return times, (leader_x, leader_y), (cf2_x, cf2_y), (cf3_x, cf3_y), (cf4_x, cf4_y)

def compute_velocity(times, x, y):
    vx = np.diff(x) / np.diff(times)
    vy = np.diff(y) / np.diff(times)
    speed = np.sqrt(vx**2 + vy**2)

    # # Clip velocities to a maximum of 1 m/s
    # speed = np.clip(speed, 0, 0.5)

    # # Normalize speed between 0 and 1
    # if len(speed) > 0:
    #     speed = (speed - np.min(speed)) / (np.max(speed) - np.min(speed) + 1e-6)  # Adding small value to avoid division by zero
    speed = apply_exponential_smoothing(speed)
    speed = np.clip(speed, 0, 0.7)
    return speed

def apply_exponential_smoothing(data, alpha=0.007):
    smoothed_data = [data[0]]  # Initialize with the first data point

    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[-1]
        smoothed_data.append(smoothed_value)
    
    return smoothed_data

def plot_trajectories(times, leader, cf2, cf3, cf4, goals, obstacles):
    # Filter the data to start from the 2-second mark
    if len(cf2[0]) < len(times):
        last_x, last_y = cf2[0][-1], cf2[1][-1]  # Get the last available data point in cf2
        missing_len = len(times) - len(cf2[0])
        # Append the last data point to cf2
        cf2 = (cf2[0] + [last_x] * missing_len, cf2[1] + [last_y] * missing_len)
    valid_indices = [i for i, time in enumerate(times) if time >= 2]
    filtered_times = [times[i] for i in valid_indices]
    filtered_leader = ([leader[0][i] for i in valid_indices], [leader[1][i] for i in valid_indices])
    filtered_cf2 = ([cf2[0][i] for i in valid_indices], [cf2[1][i] for i in valid_indices])
    filtered_cf3 = ([cf3[0][i] for i in valid_indices], [cf3[1][i] for i in valid_indices])
    # filtered_cf4 = ([cf4[0][i] for i in valid_indices], [cf4[1][i] for i in valid_indices])

    leader_speed = compute_velocity(filtered_times, filtered_leader[0], filtered_leader[1])
    cf2_speed = compute_velocity(filtered_times, filtered_cf2[0], filtered_cf2[1])
    cf3_speed = compute_velocity(filtered_times, filtered_cf3[0], filtered_cf3[1])
    # cf4_speed = compute_velocity(filtered_times, filtered_cf4[0], filtered_cf4[1])

    plt.figure(figsize=(9, 6))

    # Plot leader drone's trajectory as a dotted line
    plt.plot(filtered_leader[0], filtered_leader[1], c='black', linestyle='-', linewidth=2, label='Leader Drone')
    plt.plot([], [], c='blue', linestyle='--', label='Follower Drones', linewidth=2)
    plt.plot(filtered_cf2[0], filtered_cf2[1], c='blue', linestyle='--', linewidth=1)
    plt.plot(filtered_cf3[0], filtered_cf3[1], c='blue', linestyle='--', linewidth=1)

    # Plot CF2, CF3 trajectories with color maps
    sc2 = plt.scatter(filtered_cf2[0][1:], filtered_cf2[1][1:], c=cf2_speed, cmap='viridis', marker='o', s=20)
    sc3 = plt.scatter(filtered_cf3[0][1:], filtered_cf3[1][1:], c=cf3_speed, cmap='viridis', marker='o', s=20)


    # Plot goals (multiple goals can be plotted as scatter points)
    plt.scatter([], [], c='red', marker='x', s=120, label='Goal')
    for goal in goals:
        plt.scatter(goal[0], goal[1], c='red', marker='x', s=120)

    # Plot obstacles (multiple obstacles can be plotted as scatter points)
    plt.scatter([], [], c='brown', marker='o', s=120, label='Soft Obstacle')
    for obstacle in obstacles:
        plt.scatter(obstacle[0], obstacle[1], c='brown', marker='o', s=120)

    # Add colorbars for CFs
    plt.colorbar(sc2, label="Speed (m/s)")

    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

# Example usage
filename = 'vicon_data/exp5_3_drone_human_v2/drone_poses_exp1.txt'  # Change this to the actual filename
times, leader, cf2, cf3, cf4 = extract_positions(filename)

# Define the goals and obstacles (you can have more than one of each)
goals = [[1.691615343093872, -1.3696720600128174]]  # Example: 1 goal
obstacles = [
    [-0.43929613,  1.02555847],  
    [-0.88138467, -0.60428303],
    [0.05844557, -1.07022846],
    [0.61765647,  0.15728724]
]

plot_trajectories(times, leader, cf2, cf3, cf4, goals, obstacles)
