import time
import yaml
import random
from argparse import ArgumentParser
import numpy as np
from pathlib import Path

parser = ArgumentParser(description="Genetic Algorithm for TSP")
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

def read_tsp_data(filename):
    """
    Reads a TSP data file and returns the coordinates of the cities.

    Args:
        filename (str): The path to the TSP data file.

    Returns:
        dict: A dictionary containing the TSP data, including:
            - 'NAME': The name of the TSP instance.
            - 'TYPE': The type of the problem (e.g., 'TSP').
            - 'DIMENSION': The number of cities.
            - 'EDGE_WEIGHT_TYPE': The type of edge weights (e.g., 'EUC_2D').
            - 'NODE_COORD_SECTION': A numpy array of shape (n, 2) with the coordinates of the cities.
    """
    data = {
        'NAME': None,
        'TYPE': None,
        'DIMENSION': None,
        'EDGE_WEIGHT_TYPE': None,
        'NODE_COORD_SECTION': None
    }

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('NAME'):
                data['NAME'] = line.split(':')[1].strip()
            elif line.startswith('TYPE'):
                data['TYPE'] = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                data['DIMENSION'] = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                data['EDGE_WEIGHT_TYPE'] = line.split(':')[1].strip()
            elif line.startswith('NODE_COORD_SECTION'):
                break

        # Read the coordinates
        coords = []
        for line in file:
            if line.strip() == 'EOF':
                break
            coords.append(list(map(float, line.split()[1:])))
        coords = np.array(coords)
        data['NODE_COORD_SECTION'] = coords

    return data

def euc_2D_distance(coord1, coord2):
    """
    Computes the Euclidean distance between two points in 2D space.

    Args:
        coord1 (np.ndarray): The coordinates of the first point (x1, y1).
        coord2 (np.ndarray): The coordinates of the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(coord1 - coord2)

def compute_distance_matrix(coords):
    """
    Computes the distance matrix for a set of coordinates.

    Args:
        coords (np.ndarray): A numpy array of shape (n, 2) with the coordinates of the cities.

    Returns:
        np.ndarray: A 2D numpy array representing the distance matrix.
    """
    n = coords.shape[0]
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = euc_2D_distance(coords[i], coords[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def calculate_distance(path, distance_matrix):
    """
    Calculates the total distance of a given path.

    Args:
        path (list): A list representing the order of cities in the path.
        distance_matrix (np.ndarray): The distance matrix of the cities.

    Returns:
        float: The total distance of the path.
    """
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    total_distance += distance_matrix[path[-1], path[0]]
    return total_distance

def selection(distance_matrix, population, method='tournament', tournament_size=5):
    """
    Selects a parent from the population using the specified method.

    Args:
        distance_matrix (np.ndarray): The distance matrix of the cities.
        population (list): The current population of paths.
        method (str): The selection method ('tournament', 'roulette').
        tournament_size (int): The size of the tournament for selection.

    Returns:
        list: The selected parent path.
    """
    if method == 'tournament':
        tournament = random.sample(population, tournament_size)
        best_path = min(tournament, key=lambda path: calculate_distance(path, distance_matrix))
        return best_path
    elif method == 'roulette':
        fitness = [1 / calculate_distance(path, distance_matrix) for path in population]
        total_fitness = np.sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        return np.random.choice(population, p=probabilities)
    else:
        raise ValueError("Invalid selection method. Use 'tournament' or 'roulette'.")

def crossover(parent1, parent2, method='order'):
    """
    Performs crossover between two parents to create two children.

    Args:
        parent1 (list): The first parent path.
        parent2 (list): The second parent path.
        method (str): The crossover method ('order', 'pmx').

    Returns:
        tuple: Two children paths created
    """
    if method == 'order':
        size = len(parent1)
        start = np.random.randint(0, size)
        end = np.random.randint(start + 1, size + 1)

        child1, child2 = [None] * size, [None] * size

        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        def fill_child(child, parent):
            child_pos, parent_pos = end % size, end % size
            while None in child:
                if parent[parent_pos] not in child:
                    child[child_pos] = parent[parent_pos]
                    child_pos = (child_pos + 1) % size
                parent_pos = (parent_pos + 1) % size
            return child
        child1 = fill_child(child1, parent2)
        child2 = fill_child(child2, parent1)

        return child1, child2
    elif method == 'pmx':
        pass
    else:
        raise ValueError("Invalid crossover method. Use 'order' or 'pmx'.")
    
def mutate(path, method='swap'):
    """
    Mutates a path using the specified method.

    Args:
        path (list): The path to mutate.
        method (str): The mutation method ('swap', 'invert').

    Returns:
        list: The mutated path.
    """
    if method == 'swap':
        idx1, idx2 = np.random.choice(len(path), 2, replace=False)
        path[idx1], path[idx2] = path[idx2], path[idx1]
    elif method == 'invert':
        start, end = np.random.choice(len(path), 2, replace=False)
        if start > end:
            start, end = end, start
        path[start:end + 1] = reversed(path[start:end + 1])
    else:
        raise ValueError("Invalid mutation method. Use 'swap' or 'invert'.")
    
    return path

def genetic_tsp(
    distance_matrix,
    n,
    population_size=100,
    generation_count=1000,
    tournament_size=5,
    elite_size=5,
    crossover_rate=0.9,
    mutation_rate=0.01,
    selection_method='tournament',
    crossover_method='order',
    mutation_method='swap'
):
    """
    Genetic TSP algorithm.

    Args:
        distance_matrix (np.ndarray): The distance matrix of the cities.
        n (int): The number of cities.
        population_size (int): The size of the population.
        generation_count (int): The number of generations to run.
        tournament_size (int): The size of the tournament for selection, None for no tournament.
        elite_size (int): The number of elite individuals to keep.
        crossover_rate (float): The probability of crossover.
        mutation_rate (float): The probability of mutation.
        selection_method (str): The method for selection ('tournament', 'roulette').
        crossover_method (str): The method for crossover ('order', 'pmx').
        mutation_method (str): The method for mutation ('swap', 'invert').

    Returns:
        list: A list representing the best path found.
    """
    # Check configurations
    assert (generation_count - elite_size) % 2 == 0, "The number of generations must be even after removing elite individuals."

    # Initialize the population
    population = [list(range(n))] * population_size
    for i in range(population_size):
        np.random.shuffle(population[i])
    # if args.debug:
        # print(f"Initial population: {population}")
    
    # Initialize the best path and distance
    best_path = min(population, key=lambda path: calculate_distance(path, distance_matrix))
    best_distance = calculate_distance(best_path, distance_matrix)
    logs = [{
        'generation': 0,
        'cur_best_path': best_path,
        'cur_best_distance': best_distance,
        'best_path': best_path,
        'best_distance': best_distance
    }]
    if args.debug:
        print(f"Initial best path: {best_path}, distance: {best_distance}")

    # Main loop
    for generation in range(generation_count):
        new_population = []
        sorted_population = sorted(population, key=lambda path: calculate_distance(path, distance_matrix))

        # Elitism
        new_population.extend(sorted_population[:elite_size])

        # Crossover and mutation
        while len(new_population) < population_size:
            parent1 = selection(distance_matrix, population, selection_method, tournament_size)
            parent2 = selection(distance_matrix, population, selection_method, tournament_size)

            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, crossover_method)
            else:
                child1, child2 = parent1[:], parent2[:]

            if np.random.rand() < mutation_rate:
                mutate(child1, mutation_method)
            if np.random.rand() < mutation_rate:
                mutate(child2, mutation_method)

            new_population.append(child1)
            new_population.append(child2)

        population = new_population

        # Update the best path and distance
        cur_best_path = min(population, key=lambda path: calculate_distance(path, distance_matrix))
        cur_best_distance = calculate_distance(cur_best_path, distance_matrix)
        if cur_best_distance < best_distance:
            best_path = cur_best_path
            best_distance = cur_best_distance

        logs.append({
            'generation': generation + 1,
            'cur_best_path': cur_best_path,
            'cur_best_distance': cur_best_distance,
            'best_path': best_path,
            'best_distance': best_distance
        })
        if args.debug and generation % 100 == 0:
            print(f"Generation {generation + 1}: Best distance: {best_distance}")
    
    return best_path, best_distance, logs

def main():
    """
    Main function to execute the TSP data reading and distance matrix computation.
    """
    # Set the ROOT_DIR to the directory of the current script
    start_time = time.time()
    ROOT_DIR = Path(__file__).resolve(strict=True).parent

    # Read config from YAML file
    with open(ROOT_DIR.joinpath('config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    task_id = config.get('task_id', 0)
    data_dir = config.get('data_dir', 'data')
    tsp_list = config.get('tsp_instances')
    if not tsp_list:
        raise ValueError("No TSP instances provided in the configuration.")
    seed = config.get('seed', 42)
    population_size = config.get('population_size', 100)
    generation_count = config.get('generation_count', 1000)
    tournament_size = config.get('tournament_size', 5)
    elite_size = config.get('elite_size', 5)
    crossover_rate = config.get('crossover_rate', 0.9)
    mutation_rate = config.get('mutation_rate', 0.01)
    selection_method = config.get('selection_method', 'tournament')
    crossover_method = config.get('crossover_method', 'order')
    mutation_method = config.get('mutation_method', 'swap')

    # Set the random seed for reproducibility
    # random.seed(seed)
    # np.random.seed(seed+1)

    # Define the path to the TSP file
    tsp_file_path = ROOT_DIR / data_dir / f'{tsp_list[task_id]}.tsp'

    # Read the TSP data
    tsp_data = read_tsp_data(tsp_file_path)

    # Print the Running environment
    print("Running with:")
    print(f"\nTSP instance: {tsp_file_path}")
    print(f"TSP Instance Name: {tsp_data['NAME']}")
    print(f"Number of Cities: {tsp_data['DIMENSION']}")
    print(f"Edge Weight Type: {tsp_data['EDGE_WEIGHT_TYPE']}")
    print("\nConfiguration:")
    print(f"Population Size: {population_size}")
    print(f"Generation Count: {generation_count}")
    print(f"Tournament Size: {tournament_size}")
    print(f"Elite Size: {elite_size}")
    print(f"Crossover Rate: {crossover_rate}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Selection Method: {selection_method}")
    print(f"Crossover Method: {crossover_method}")
    print(f"Mutation Method: {mutation_method}")


    # Extract coordinates
    coords = tsp_data['NODE_COORD_SECTION']

    # Compute the distance matrix
    distance_matrix = compute_distance_matrix(coords)

    path, distance, logs = genetic_tsp(
        distance_matrix, tsp_data['DIMENSION'],
        population_size, generation_count,
        tournament_size, elite_size,
        crossover_rate, mutation_rate,
        selection_method, crossover_method,
        mutation_method
    )

    end_time = time.time()
    exec_time = end_time - start_time

    # Print the results
    print(f"Task: {tsp_data['NAME']}")
    print(f"Execution time: {exec_time:.2f} seconds")
    print(f"Best distance: {distance}")

    # Save the results to a YAML file
    result = {
        # 'Task': tsp_data['NAME'],
        'Configuration': {
            'Population_size': population_size,
            'Generation_count': generation_count,
            'Tournament_size': tournament_size,
            'Elite_size': elite_size,
            'Crossover_rate': crossover_rate,
            'Mutation_rate': mutation_rate,
            'Selection_method': selection_method,
            'Crossover_method': crossover_method,
            'Mutation_method': mutation_method
        },
        # 'Execution_time': exec_time,
        # 'Best_distance': distance,
        # 'Best_path': path,
    }
    # if args.debug:
        # result['Logs'] = logs

    result_file = ROOT_DIR / 'output' / f'{tsp_list[task_id]}_{population_size}_{tournament_size}_{elite_size}.yaml'
    with open(result_file, 'w') as file:
        yaml.dump(result, file, default_flow_style=False)

if __name__ == "__main__":
    main()
