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
    speed = np.clip(speed, 0, 1)

    # # Normalize speed between 0 and 1
    # if len(speed) > 0:
    #     speed = (speed - np.min(speed)) / (np.max(speed) - np.min(speed) + 1e-6)  # Adding small value to avoid division by zero
    # speed = apply_exponential_smoothing(speed)
    return speed

def apply_exponential_smoothing(data, alpha=0.2):
    smoothed_data = [data[0]]  # Initialize with the first data point

    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[-1]
        smoothed_data.append(smoothed_value)
    
    return smoothed_data

def plot_trajectories(times, leader, cf2, cf3, cf4, goals, obstacles):
    # Filter the data to start from the 2-second mark
    valid_indices = [i for i, time in enumerate(times) if time >= 2]
    filtered_times = [times[i] for i in valid_indices]
    filtered_leader = ([leader[0][i] for i in valid_indices], [leader[1][i] for i in valid_indices])
    filtered_cf2 = ([cf2[0][i] for i in valid_indices], [cf2[1][i] for i in valid_indices])
    filtered_cf3 = ([cf3[0][i] for i in valid_indices], [cf3[1][i] for i in valid_indices])
    filtered_cf4 = ([cf4[0][i] for i in valid_indices], [cf4[1][i] for i in valid_indices])

    leader_speed = compute_velocity(filtered_times, filtered_leader[0], filtered_leader[1])
    cf2_speed = compute_velocity(filtered_times, filtered_cf2[0], filtered_cf2[1])
    cf3_speed = compute_velocity(filtered_times, filtered_cf3[0], filtered_cf3[1])
    cf4_speed = compute_velocity(filtered_times, filtered_cf4[0], filtered_cf4[1])

    plt.figure(figsize=(9, 6))

    # Plot leader drone's trajectory with velocity colormap

    # Plot CF2, CF3, and CF4 trajectories with lines
    plt.plot(filtered_leader[0], filtered_leader[1], c='black', linestyle='-', label='Leader Drone', linewidth=2)
    plt.plot([], [], c='blue', linestyle='--', label='Follower Drones', linewidth=2)
    plt.plot(filtered_cf2[0], filtered_cf2[1], c='blue', linestyle='--', linewidth=1)
    plt.plot(filtered_cf3[0], filtered_cf3[1], c='blue', linestyle='--', linewidth=1)
    # plt.plot(filtered_cf4[0], filtered_cf4[1], c='blue', linestyle='--', linewidth=1)

    # Plot CF2, CF3, and CF4 trajectories with color maps
    sc2 = plt.scatter(filtered_cf2[0][1:], filtered_cf2[1][1:], c=cf2_speed, cmap='viridis', marker='o', s=20)
    sc3 = plt.scatter(filtered_cf3[0][1:], filtered_cf3[1][1:], c=cf3_speed, cmap='viridis', marker='o', s=20)
    # sc4 = plt.scatter(filtered_cf4[0][1:], filtered_cf4[1][1:], c=cf3_speed, cmap='viridis', marker='o', s=20)

    # Plot goals (multiple goals can be plotted as scatter points)
    plt.scatter([], [], c='red', marker='x', s=120, label='Moving Goal')
    for i, goal in enumerate(goals):
        alpha_value = 1.0 if i == len(goals) - 1 else 0.6  # Set alpha to 1 for final goal, 0.5 for others
        plt.scatter(goal[0], goal[1], c='red', marker='x', s=120, alpha=alpha_value)

    # Plot obstacles (multiple obstacles can be plotted as scatter points)
    plt.scatter([], [], c='orange', marker='o', s=120, label='Hard Obstacle')
    for obstacle in obstacles:
        plt.scatter(obstacle[0], obstacle[1], c='orange', marker='o', s=120)

    plt.colorbar(sc2, label="Speed (m/s)")

    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

def plot_velocities(times, cf2, cf3, cf4, alpha=0.1):
    # Compute velocities for CF2, CF3, CF4
    cf2_velocity = compute_velocity(times, cf2[0], cf2[1])
    cf3_velocity = compute_velocity(times, cf3[0], cf3[1])
    cf4_velocity = compute_velocity(times, cf4[0], cf4[1])

    # Apply exponential smoothing
    cf2_velocity = apply_exponential_smoothing(cf2_velocity, alpha)
    cf3_velocity = apply_exponential_smoothing(cf3_velocity, alpha)
    cf4_velocity = apply_exponential_smoothing(cf4_velocity, alpha)

    plt.figure(figsize=(8, 6))

    # Plot velocities of CF2, CF3, CF4
    plt.plot(times[1:], cf2_velocity, c='blue', label='CF2 Velocity', linewidth=2)
    plt.plot(times[1:], cf3_velocity, c='purple', label='CF3 Velocity', linewidth=2)
    plt.plot(times[1:], cf4_velocity, c='brown', label='CF4 Velocity', linewidth=2)

    # Set plot labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocities of Follower Drones (CF2, CF3, CF4)')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
filename = 'vicon_data/exp4_4_drone_dynamic/drone_poses_exp1.txt'  # Change this to the actual filename
times, leader, cf2, cf3, cf4 = extract_positions(filename)

# Define the goals and obstacles (you can have more than one of each)
goals = [[1.579100251197815, -1.4502214193344116],
         [1.5091170072555542, -1.5285334587097168],
         [1.3990601301193237, -1.6133846044540405],
         [1.2630541324615479, -1.6930264234542847],
         [1.023791790008545, -1.8537235260009766]]  # Example: 1 goal
obstacles = [
    [-0.32883504,  0.17331941],  
    [-0.6948629,  -0.05946442],
    [1.36668813, -0.18934891]
]

plot_trajectories(times, leader, cf2, cf3, cf4, goals, obstacles)
# plot_velocities(times, cf2, cf3, cf4)
