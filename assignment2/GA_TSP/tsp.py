import time
import yaml
import json
import random
from argparse import ArgumentParser
import numpy as np
from pathlib import Path

parser = ArgumentParser(description="Genetic Algorithm for TSP")
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--best', action='store_true', help='Enable best mode')
parser.add_argument('--ver', type=str, default='default', help='Version of the configuration to use')
parser.add_argument('--log', action='store_true', help='Enable log mode')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
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
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    total_distance += distance_matrix[path[-1], path[0]]
    return total_distance

def selection(distance_matrix, population, cfg):
    """
    Selects a parent from the population using the specified method.

    Args:
        distance_matrix (np.ndarray): The distance matrix of the cities.
        population (list): The current population of paths.
        cfg (dict): Configuration dictionary containing the parameters for the algorithm.
            method (str): The selection method ('tournament', 'roulette').
            tournament_size (int): The size of the tournament for selection.

    Returns:
        list: The selected parent path.
    """
    method = cfg['selection_method']
    if method == 'tournament':
        tournament_size = cfg['tournament_size']
        tournament = random.sample(population, tournament_size)
        best_path = min(tournament, key=lambda path: calculate_distance(path, distance_matrix))
        return best_path
    elif method == 'roulette':
        fitness = [1 / calculate_distance(path, distance_matrix) for path in population]
        total_fitness = np.sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        selected_idx = np.random.choice(np.arange(len(population)), p=probabilities)
        return population[selected_idx]
    elif method == 'adaptive_tournament':
        adapt_cfg = cfg['adaptive_tournament']
        max_size = adapt_cfg['max_size']
        min_size = adapt_cfg['min_size']
        tournament_size = max(min_size, int(max_size * (1 - cfg['generation'] / cfg['generation_count'])))
        tournament = random.sample(population, tournament_size)
        best_path = min(tournament, key=lambda path: calculate_distance(path, distance_matrix))
        return best_path
    else:
        raise ValueError("Invalid selection method. Use 'tournament' or 'roulette'.")

def crossover(parent1, parent2, method='order'):
    """
    Performs crossover between two parents to create two children.

    Args:
        parent1 (list): The first parent path.
        parent2 (list): The second parent path.
        method (str): The crossover method ('order').

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
    else:
        raise ValueError("Invalid crossover method. Use 'order'.")

def get_mutate(mutation_rate, generation, logs, cfg):
    """
    Gets the mutation rate based on the current generation and logs.
    Args:
        generation (int): The current generation number.
        logs (list): The logs of the generations.
        cfg (dict): Configuration dictionary containing the parameters for the algorithm.
    """
    if cfg['mutation_rate_method'] == 'adaptive':
        adapt_cfg = cfg['adaptive_mutation']
        if len(logs) < 10: 
            return mutation_rate
        improvement = logs[-10]['cur_best_distance'] - logs[-1]['cur_best_distance']
        improvement_rate = improvement / logs[-5]['cur_best_distance']
        if improvement_rate > adapt_cfg['decay_threshold']:
            mutation_rate = max(mutation_rate * adapt_cfg['decay_rate'], 0.001)
        elif improvement_rate < adapt_cfg['increase_threshold']:
            mutation_rate = min(mutation_rate * adapt_cfg['increase_rate'], 0.6)
        return mutation_rate
    elif cfg['mutation_rate_method'] == 'fixed':
        return mutation_rate
    elif cfg['mutation_rate_method'] == 'linear':
        max_generation = cfg['generation_count']
        if generation % 100 == 0:
            mutation_rate = cfg['mutation_rate'] * (1 - generation / max_generation)
        return max(mutation_rate, 0.01)
    else:
        raise ValueError("Invalid mutation rate method.")

def mutate(generation, cfg, path, method='swap'):
    """
    Mutates a path using the specified method.

    Args:
        generation (int): The current generation number.
        path (list): The path to mutate.
        cfg (dict): Configuration dictionary containing the parameters for the algorithm.
        method (str): The mutation method ('swap', 'invert', 'adaptive').

    Returns:
        list: The mutated path.
    """
    size = len(path)
    if method == 'swap':
        idx1, idx2 = np.random.choice(size, 2, replace=False)
        path[idx1], path[idx2] = path[idx2], path[idx1]
    elif method == 'invert':
        start = np.random.randint(0, size)
        end = np.random.randint(start + 1, size + 1)
        path[start:end] = reversed(path[start:end])
    elif method == 'adaptive':
        if generation / cfg['generation_count'] > 0.8:
            return mutate(generation, cfg, path, 'swap')
        else:
            return mutate(generation, cfg, path, 'invert')
    else:
        raise ValueError("Invalid mutation method. Use 'swap' or 'invert'.")
    
    return path

def init_population(coords, n, cfg):
    """
    Initializes a population of paths.

    Args:
        coords (np.ndarray): The coordinates of the cities.
        n (int): The number of cities.
        cfg (dict): Configuration dictionary containing the parameters for the algorithm.
            init_method (str): The method for initializing the population ('random', 'greedy').
            population_size (int): The size of the population.

    Returns:
        list: A list of paths representing the initial population.
    """
    population_size = cfg['population_size']
    if cfg['init_method'] == 'random':
        population = [random.sample(range(n), n) for _ in range(population_size)]
    elif cfg['init_method'] == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans_cfg = cfg['init_kmeans']
        n_clusters = kmeans_cfg['n_clusters']
        kmeans = KMeans(n_clusters=n_clusters, max_iter=kmeans_cfg['max_iter'], random_state=cfg['seed']+2)
        kmeans.fit(coords)
        labels = kmeans.labels_
        population = []
        for i in range(population_size):
            route = []
            for j in random.sample(range(n_clusters), n_clusters):
                cluster_indices = np.where(labels == j)[0]
                if len(cluster_indices) > 0:
                    np.random.shuffle(cluster_indices)
                    route.extend(cluster_indices.tolist())
            population.append(route)
    else:
        raise ValueError("Invalid initialization method. Use 'random' or 'kmeans'.")
    return population


def get_crossover_rate(crossover_rate, generation, cfg):
    """
    Gets the crossover rate based on the current generation.
    
    Args:
        crossover_rate (float): The current crossover rate.
        generation (int): The current generation number.
        cfg (dict): Configuration dictionary containing the parameters for the algorithm.
    """
    if cfg['crossover_rate_method'] == 'linear':
        max_generation = cfg['generation_count']
        if generation % 100 == 0:
            crossover_rate = 0.6 + (cfg['crossover_rate'] - 0.6) * (1 - generation / max_generation)
        return max(crossover_rate, 0.6)
    elif cfg['crossover_rate_method'] == 'fixed':
        return crossover_rate
    else:
        raise ValueError("Invalid crossover rate method.")

def genetic_tsp(coords, distance_matrix, n, cfg):
    """
    Genetic TSP algorithm.

    Args:
        coords (np.ndarray): The coordinates of the cities.
        distance_matrix (np.ndarray): The distance matrix of the cities.
        n (int): The number of cities.
        cfg (dict): Configuration dictionary containing the parameters for the algorithm.
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
    generation_count = cfg['generation_count']
    population_size = cfg['population_size']
    elite_size = cfg['elite_size']
    crossover_rate = cfg['crossover_rate']
    mutation_rate = cfg['mutation_rate']
    # Check configurations
    assert (generation_count - elite_size) % 2 == 0, "The number of generations must be even after removing elite individuals."

    # Initialize the population
    population = init_population(coords, n, cfg)
    
    # Initialize the best path and distance
    best_path = min(population, key=lambda path: calculate_distance(path, distance_matrix))
    best_distance = calculate_distance(best_path, distance_matrix)
    logs = [{
        'generation': 0,
        'cur_best_distance': best_distance,
        'best_distance': best_distance,
        'mutation_rate': mutation_rate,
    }]
    if args.debug:
        print(f"Initial best path: {best_path}, distance: {best_distance}")

    # Main loop
    for generation in range(generation_count):
        cfg['generation'] = generation
        new_population = []
        sorted_population = sorted(population, key=lambda path: calculate_distance(path, distance_matrix))

        # Elitism
        new_population.extend(sorted_population[:elite_size])

        # Mutation rate and Crossover rate adjustment
        mutation_rate = get_mutate(mutation_rate, generation, logs, cfg)
        crossover_rate = get_crossover_rate(crossover_rate, generation, cfg)

        # Crossover and mutation
        while len(new_population) < population_size:
            parent1 = selection(distance_matrix, population, cfg)
            parent2 = selection(distance_matrix, population, cfg)

            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, cfg['crossover_method'])
            else:
                child1, child2 = parent1[:], parent2[:]

            if np.random.rand() < mutation_rate:
                mutate(generation, cfg, child1, cfg['mutation_method'])
            if np.random.rand() < mutation_rate:
                mutate(generation, cfg, child2, cfg['mutation_method'])

            new_population.append(child1)
            new_population.append(child2)

        population = new_population

        # Update the best path and distance
        cur_best_path = min(population, key=lambda path: calculate_distance(path, distance_matrix))
        cur_best_distance = calculate_distance(cur_best_path, distance_matrix)
        if cur_best_distance < best_distance:
            best_path = cur_best_path.copy()
            best_distance = cur_best_distance

        logs.append({
            'generation': generation + 1,
            'cur_best_distance': cur_best_distance,
            'best_distance': best_distance,
            'mutation_rate': mutation_rate
        })
        if args.debug and generation % 50 == 0:
            print(f"Generation {generation + 1}: Best distance: {int(best_distance)}, Current Best dis: {int(cur_best_distance)}, Mutation rate: {mutation_rate}, Crossover rate: {crossover_rate}")
    
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
    
    tsp_instance = tsp_list[task_id]

    if args.best:
        cfg = config.get(f'{tsp_instance}_best', config['default'])
    else:
        cfg = config.get(args.ver, config['default'])

    # Set the random seed for reproducibility
    if args.seed:
        cfg['seed'] = args.seed
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'] + 1)

    # Define the path to the TSP file
    tsp_file_path = ROOT_DIR / data_dir / f'{tsp_instance}.tsp'

    # Read the TSP data
    tsp_data = read_tsp_data(tsp_file_path)

    # Print the Running environment
    print("Running with:")
    print(f"\nTSP instance: {tsp_file_path}")
    print(f"TSP Instance Name: {tsp_data['NAME']}")
    print(f"Number of Cities: {tsp_data['DIMENSION']}")
    print(f"Edge Weight Type: {tsp_data['EDGE_WEIGHT_TYPE']}")
    print("\nConfiguration:")
    print(f"Seed: {cfg['seed']}")
    print(f"Initialization Method: {cfg['init_method']}")
    if cfg['init_method'] == 'kmeans':
        print(f"KMeans Clusters: {cfg['init_kmeans']['n_clusters']}")
        print(f"KMeans Max Iterations: {cfg['init_kmeans']['max_iter']}")
    print(f"Population Size: {cfg['population_size']}")
    print(f"Generation Count: {cfg['generation_count']}")
    print(f"Tournament Size: {cfg['tournament_size']}")
    print(f"Elite Size: {cfg['elite_size']}")
    print(f"Crossover Rate: {cfg['crossover_rate']}")
    print(f"Crossover Rate Method: {cfg['crossover_rate_method']}")
    print(f"Mutation Rate: {cfg['mutation_rate']}")
    print(f"Mutation Rate Method: {cfg['mutation_rate_method']}")
    print(f"Selection Method: {cfg['selection_method']}")
    print(f"Crossover Method: {cfg['crossover_method']}")
    print(f"Mutation Method: {cfg['mutation_method']}")


    # Extract coordinates
    coords = tsp_data['NODE_COORD_SECTION']

    # Compute the distance matrix
    distance_matrix = compute_distance_matrix(coords)

    path, distance, logs = genetic_tsp(coords, distance_matrix, tsp_data['DIMENSION'], cfg)

    end_time = time.time()
    exec_time = end_time - start_time

    # Print the results
    print(f"Task: {tsp_data['NAME']}")
    print(f"Execution time: {exec_time:.2f} seconds")
    print(f"Best distance: {distance:.0f}")

    # Save the results to a json file
    result = {
        'Task': tsp_data['NAME'],
        'Configuration': cfg,
        'Execution_time': exec_time,
        'Best_distance': distance,
        'Best_path': path,
    }
    if args.debug:
        result['Logs'] = logs


    result_path = ROOT_DIR / 'output'
    result_path.mkdir(parents=True, exist_ok=True)
    result_name = f'{tsp_instance}_{cfg["seed"]}_{cfg["population_size"]}_{cfg["tournament_size"]}_{cfg["elite_size"]}_{cfg["crossover_rate"]}_{cfg["mutation_rate"]}_{cfg["mutation_method"]}'
    if args.best:
        result_name = f'{tsp_instance}_best'
    
    if args.log:
        result_file = result_path / f'{result_name}.json'
        with open(result_file, 'w') as file:        
            json.dump(result, file, indent=4)
        print(f"Results saved to {result_file}")

    # Plot the path
    plot_path(coords, path, tsp_instance, distance, result_path / f'{result_name}.png')
    plot_(logs, 'cur_best_distance', result_path / f'distance_{result_name}.png')
    plot_(logs, 'mutation_rate', result_path / f'mutation_{result_name}.png')


def plot_path(coords, path, tsp_instance, distance, output_file):
    """
    Plots the path on a 2D graph.

    Args:
        coords (np.ndarray): The coordinates of the cities.
        path (list): The order of cities in the path.
        tsp_instance (str): The name of the TSP instance.
        distance (float): The total distance of the path.
        output_file (str): The path to save the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(coords[path, 0], coords[path, 1], 'o-', markersize=3)
    plt.plot([coords[path[-1], 0], coords[path[0], 0]], [coords[path[-1], 1], coords[path[0], 1]], 'o-')
    plt.title('TSP Instance: {}, Distance: {:.0f}'.format(tsp_instance, distance))
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.savefig(output_file)
    # plt.show()
    plt.close()
    print(f"Path plot saved to {output_file}")

def plot_(logs, name, output_file):
    """
    Plots the distances over generations.

    Args:
        logs (list): The logs of the generations.
        output_file (str): The path to save the plot.
    """
    import matplotlib.pyplot as plt

    generations = [log['generation'] for log in logs]
    distances = [log[name] for log in logs]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, distances)
    plt.title(f'{name} over Generations')
    plt.xlabel('Generation')
    plt.ylabel(name)
    plt.grid()
    plt.savefig(output_file)
    # plt.show()
    plt.close()
    print(f"{name} plot saved to {output_file}")

if __name__ == "__main__":
    main()
