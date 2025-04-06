import numpy as np
import time
from pathlib import Path
import heapq

class Node:
    target = tuple([
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 0)
    ])

    directions = [
        (0, 1),  # right
        (1, 0),  # down
        (0, -1), # left
        (-1, 0)  # up
    ]

    def __init__(self,
            state: list,
            parent: 'Node' = None,
            gs: int = -1,
            direction: int = -1,
            number: int = 0):
        self.state = tuple(tuple(row) for row in state)
        self.parent = parent
        self.gs = gs
        self.direction = direction
        self.number = number
        self.hs = self.hScore()
        self.fs = self.gs + self.hs
        self._hash = hash(self.state)

    def hScore(self) -> int:
        distance = 0
        # Manhattan Distance Heuristic
        for i in range(4):
            for j in range(4):
                val = self.state[i][j]
                if val == 0:
                    continue
                target_x = (val - 1) // 4
                target_y = (val - 1) % 4
                distance += abs(i - target_x) + abs(j - target_y)
        # Linear Conflict Heuristic
        for i in range(4):
            row = [self.state[i][j] - 1 for j in range(4) if self.state[i][j] != 0]
            for j in range(len(row) - 1):
                for k in range(j + 1, len(row)):
                    if (row[j] // 4 == i) and (row[k] // 4 == i) and (row[j] % 4 > row[k] % 4):
                        distance += 2
            col = [self.state[j][i] - 1 for j in range(4) if self.state[j][i] != 0]
            for j in range(len(col) - 1):
                for k in range(j + 1, len(col)):
                    if (col[j] % 4 == i) and (col[k] % 4 == i) and (col[j] // 4 > col[k] // 4):
                        distance += 2
        return distance

    def neighbors(self):
        """
        Generate all possible states by moving the blank space
        Returns:
            generator of Node objects representing the neighbors
        """
        x, y = next((i, j) for i in range(4) for j in range(4) if self.state[i][j] == 0)
        for dir, (dx, dy) in enumerate(self.directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4:
                new_state = [list(row) for row in self.state]
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                yield Node(new_state, self, self.gs + 1, dir, self.state[nx][ny])

    def __lt__(self, other: 'Node') -> bool:
        return self.fs < other.fs
    
    def __repr__(self) -> str:
        return "\n".join([" ".join(map(str, row)) for row in self.state])

    def __eq__(self, other: 'Node') -> bool:
        return self.state == other.state

    def __hash__(self) -> int:
        return self._hash
    

def a_star(start_state: list) -> list:
    """
    Implements the A* algorithm to find the shortest path to the target state.
    Args:
        start (list of list): The initial state of the puzzle.
    Returns:
        list: The sequence of states leading to the solution.
    """
    start_node = Node(start_state)
    open_set = []
    heapq.heappush(open_set, start_node)
    gScore = {start_node: 0}
    close_set = set()

    while open_set:
        cur = heapq.heappop(open_set)
        close_set.add(cur)
        if cur.state == Node.target:
            return reconstruct_path(cur), len(close_set)
        for neighbor in cur.neighbors():
            if neighbor in close_set:
                continue
            if neighbor not in gScore or neighbor.gs < gScore[neighbor]:
                gScore[neighbor] = neighbor.gs
                heapq.heappush(open_set, neighbor)
    return None

def reconstruct_path(cur):
    total_path = []
    number_path = []
    while cur is not None:
        total_path.append(cur)
        number_path.append(cur.number)
        cur = cur.parent
    return total_path[::-1], number_path[:-1][::-1]


def run(test_id: int):
    print(f"\nRunning test {test_id}...")
    start_time = time.time()
    (path, number_path), visited_nodes = a_star(start_state[test_id])
    end_time = time.time()

    save_path = Path(__file__).resolve().parent
    save_file = save_path.joinpath(f'output_{test_id}_LC.txt')
    with open(save_file, "w") as f:
        dir = ["Right", "Down", "Left", "Up", "Start"]
        f.write(f"Time taken: {end_time - start_time:.2f} seconds\n")
        f.write(f"Visited nodes: {visited_nodes}\n")
        dir_brief = ["R", "D", "L", "U", "S"]
        path_str = ''
        for i, node in enumerate(path):
            path_str += dir_brief[node.direction]
        f.write(f"Directions: {path_str}\n")
        f.write(f"Path: {number_path}\n")
        f.write(f"Path length: {len(path) - 1}\n\n")
        for i, node in enumerate(path):
            f.write(f"Step {i}, {dir[node.direction]}:\n")
            f.write(repr(node))
            f.write("\n\n")

    print(f"Test {test_id} completed.")
    print(f"Number of steps: {len(path) - 1}")
    print(f"Visited nodes: {visited_nodes}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Output saved to:", save_file)

if __name__ == "__main__":
    start_time = time.time()
    
    start_state = [
        [[1, 2, 4, 8], [5, 7, 11, 10], [13, 15, 0, 3], [14, 6, 9, 12]],
        [[14, 10, 6, 0], [4, 9, 1, 8], [2, 3, 5, 11], [12, 13, 7, 15]],
        [[5, 1, 3, 4], [2, 7, 8, 12], [9, 6, 11, 15], [0, 13, 10, 14]],
        [[6, 10, 3, 15], [14, 8, 7, 11], [5, 1, 0, 2], [13, 12, 9, 4]], 
        [[11, 3, 1, 7], [4, 6, 8, 2], [15, 9, 10, 13], [14, 12, 5, 0]],
        [[0, 5, 15, 14], [7, 9, 6, 13], [1, 2, 12, 10], [8, 11, 4, 3]]
    ]

    for test_id in range(0, 6):
        run(test_id)
    # run(5)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
