import numpy as np
import matplotlib.pyplot as plt
import random

# Hector David Peralta Ramirez
# Prague City University
# Programming Fundamentals

# States for the regeneration process
states = ["No regeneration", "Initiation of regeneration", "Partial regeneration", "Complete regeneration"]

# Transition matrices for the axolotl (Tail, Spinal Cord, Internal organs, Skin)
transition_axolotl_tail = np.array([[0.05, 0.15, 0.1, 0.7],
                                    [0.0, 0.05, 0.15, 0.8],
                                    [0.0, 0.0, 0.1, 0.9],
                                    [0.0, 0.0, 0.0, 1.0]])

transition_axolotl_Spine = np.array([[0.1, 0.2, 0.4, 0.3],
                                     [0.0, 0.1, 0.3, 0.6],
                                     [0.0, 0.0, 0.2, 0.8],
                                     [0.0, 0.0, 0.0, 1.0]])

transition_axolotl_organs = np.array([[0.2, 0.3, 0.3, 0.2],
                                      [0.0, 0.2, 0.4, 0.4],
                                      [0.0, 0.0, 0.3, 0.7],
                                      [0.0, 0.0, 0.0, 1.0]])

transition_axolotl_skin = np.array([[0.05, 0.15, 0.1, 0.7],
                                    [0.0, 0.05, 0.15, 0.8],
                                    [0.0, 0.0, 0.1, 0.9],
                                    [0.0, 0.0, 0.0, 1.0]])

# Transition matrices for the Humans (Scrape, fingertip, arm, liver)
transition_human_scrape = np.array([[0.1, 0.3, 0.4, 0.2],
                                    [0.0, 0.1, 0.4, 0.5],
                                    [0.0, 0.0, 0.3, 0.7],
                                    [0.0, 0.0, 0.0, 1.0]])

transition_human_fingertip = np.array([[0.3, 0.3, 0.3, 0.1],
                                       [0.0, 0.3, 0.4, 0.3],
                                       [0.0, 0.0, 0.4, 0.6],
                                       [0.0, 0.0, 0.0, 1.0]])

transition_human_arm = np.array([[0.9, 0.09, 0.01, 0.0],
                                 [0.1, 0.8, 0.1, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0]])

transition_human_liver = np.array([[0.3, 0.3, 0.3, 0.1],
                                   [0.0, 0.3, 0.4, 0.3],
                                   [0.0, 0.0, 0.4, 0.6],
                                   [0.0, 0.0, 0.0, 1.0]])

# Function to simulate the regeneration process
def simulate_regeneration_process(transition_matrix, steps):
    actual_state = 0
    sequence_states = [actual_state]
    for _ in range(steps - 1):
        actual_state = np.random.choice(range(len(transition_matrix)), p=transition_matrix[actual_state])
        sequence_states.append(actual_state)
    return sequence_states

steps_simulations = 10

# Examples of simulations for each transition matrix
example_simulation_tail = simulate_regeneration_process(transition_axolotl_tail, steps_simulations)
example_simulation_spine = simulate_regeneration_process(transition_axolotl_Spine, steps_simulations)
example_simulation_organs = simulate_regeneration_process(transition_axolotl_organs, steps_simulations)
example_simulation_skin = simulate_regeneration_process(transition_axolotl_skin, steps_simulations)

example_simulation_scrape = simulate_regeneration_process(transition_human_scrape, steps_simulations)
example_simulation_fingertip = simulate_regeneration_process(transition_human_fingertip, steps_simulations)
example_simulation_arm = simulate_regeneration_process(transition_human_arm, steps_simulations)
example_simulation_liver = simulate_regeneration_process(transition_human_liver, steps_simulations)

# Print the results
print("Regeneration Processes of Human and Axolotl\nHector David Peralta Ramirez")

print("Tail - Axolotl, Number of Steps: " + str(10))
print(example_simulation_tail)
print("Spinal Cord - Axolotl, Number of Steps: " + str(10))
print(example_simulation_spine)
print("Organs - Axolotl, Number of Steps: " + str(10))
print(example_simulation_organs)
print("Skin - Axolotl, Number of Steps: " + str(10))
print(example_simulation_skin)

print("Scrape - Human, Number of Steps: " + str(10))
print(example_simulation_scrape)
print("FingerTip - Human, Number of Steps: " + str(10))
print(example_simulation_fingertip)
print("Arm - Human, Number of Steps: " + str(10))
print(example_simulation_arm)
print("Liver - Human, Number of Steps: " + str(10))
print(example_simulation_liver)

def simulation_monte_carlo(transition_matrix, steps, num_simulations):
    final_results = []
    for _ in range(num_simulations):
        sequence = simulate_regeneration_process(transition_matrix, steps)
        final_state = sequence[-1]
        final_results.append(final_state)
    return final_results

matrices = [
    transition_axolotl_tail, transition_axolotl_Spine,
    transition_axolotl_organs, transition_axolotl_skin,
    transition_human_scrape, transition_human_fingertip,
    transition_human_arm, transition_human_liver
]

num_simulations = 10000
steps = 10

results_all_matrices = {}
for i, matrix in enumerate(matrices):
    results = simulation_monte_carlo(matrix, steps, num_simulations)
    results_all_matrices[f"matrix {i+1}"] = results

for i, (name, results) in enumerate(results_all_matrices.items()):
    average = np.mean(results)
    median = np.median(results)
    standard_deviation = np.std(results)
    print(f"Results for {name}:")
    print(f"Average of the final state: {average:.2f}")
    print(f"Median of the final state: {median}")
    print(f"Standard deviation of the final state: {standard_deviation:.2f}\n")

# Stationary distribution selection menu
matrices1 = {
    1: ("axolotl_tail", np.array([[0.05, 0.15, 0.1, 0.7],
                                  [0.0, 0.05, 0.15, 0.8],
                                  [0.0, 0.0, 0.1, 0.9],
                                  [0.0, 0.0, 0.0, 1.0]])),
    2: ("axolotl_spine", np.array([[0.1, 0.2, 0.4, 0.3],
                                   [0.0, 0.1, 0.3, 0.6],
                                   [0.0, 0.0, 0.2, 0.8],
                                   [0.0, 0.0, 0.0, 1.0]])),
    3: ("axolotl_organs", np.array([[0.2, 0.3, 0.3, 0.2],
                                    [0.0, 0.2, 0.4, 0.4],
                                    [0.0, 0.0, 0.3, 0.7],
                                    [0.0, 0.0, 0.0, 1.0]])),
    4: ("axolotl_skin", np.array([[0.05, 0.15, 0.1, 0.7],
                                  [0.0, 0.05, 0.15, 0.8],
                                  [0.0, 0.0, 0.1, 0.9],
                                  [0.0, 0.0, 0.0, 1.0]])),
    5: ("human_scrape", np.array([[0.1, 0.3, 0.4, 0.2],
                                  [0.0, 0.1, 0.4, 0.5],
                                  [0.0, 0.0, 0.3, 0.7],
                                  [0.0, 0.0, 0.0, 1.0]])),
    6: ("human_fingertip", np.array([[0.3, 0.3, 0.3, 0.1],
                                     [0.0, 0.3, 0.4, 0.3],
                                     [0.0, 0.0, 0.4, 0.6],
                                     [0.0, 0.0, 0.0, 1.0]])),
    7: ("human_arm", np.array([[0.9, 0.09, 0.01, 0.0],
                               [0.1, 0.8, 0.1, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]])),
    8: ("human_liver", np.array([[0.9, 0.09, 0.01, 0.0],
                                 [0.1, 0.8, 0.1, 0.0],
                                 [0.2, 0.1, 0.7, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]]))
}

def calculate_stationary_distribution(matrix, n_steps):
    result_matrix = np.copy(matrix)
    for _ in range(n_steps - 1):
        result_matrix = np.dot(result_matrix, matrix)
    return result_matrix

print("Please choose a matrix:")
for number, (name, _) in matrices1.items():
    print(f"{number}: {name}")

user_choice = int(input("Enter the number of your choice: "))

if user_choice in matrices1:
    _, chosen_matrix = matrices1[user_choice]
    n_steps = 10000000
    stationary_distribution = calculate_stationary_distribution(chosen_matrix, n_steps)
    print(f"Stationary distribution for {matrices1[user_choice][0]}:\n{stationary_distribution}")
else:
    print("Invalid choice.")
