import csv

# Initialize variables
mass, damping, stiffness, impedance_force_coefficient, distance_between_drones_and_virtual_leader = None, None, None, None, None

# Read the CSV file and extract values
with open("impedance_parameters.csv", mode="r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        key, value = row
        if key == "mass":
            mass = float(value)
        elif key == "damping":
            damping = float(value)
        elif key == "stiffness":
            stiffness = float(value)
        elif key == "impedance_force_coefficient":
            impedance_force_coefficient = float(value)
        elif key == "distance_between_drones_and_virtual_leader":
            distance_between_drones_and_virtual_leader = float(value)

# Print the extracted values
print(f"Extracted Values:\n"
      f"Mass: {mass}\n"
      f"Damping: {damping}\n"
      f"Stiffness: {stiffness}\n"
      f"Impedance Force Coefficient: {impedance_force_coefficient}\n"
      f"Distance Between Drones and Virtual Leader: {distance_between_drones_and_virtual_leader}")
