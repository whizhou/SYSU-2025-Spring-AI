import numpy as np
import time
from pathlib import Path
import argparse as parse

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
            gs: int,
            parent: 'Node' = None,
            direction: int = -1):
        self.state = tuple(tuple(row) for row in state)
        self.parent = parent
        self.gs = gs
        self.direction = direction
        self.hs = self.hScore()
        self.fs = self.gs + self.hs
        self._hash = hash(self.state)

    def hScore(self) -> int:
        distance = 0
        for i in range(4):
            for j in range(4):
                val = self.state[i][j]
                if val == 0:
                    continue
                target_x = (val - 1) // 4
                target_y = (val - 1) % 4
                distance += abs(i - target_x) + abs(j - target_y)
        return distance

    def neighbors(self):
        """
        Generate all possible states by moving the blank space
        Returns:
            
        """
        x, y = next((i, j) for i in range(4) for j in range(4) if self.state[i][j] == 0)
        for dir, (dx, dy) in enumerate(self.directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4:
                new_state = [list(row) for row in self.state]
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                yield Node(new_state, self.gs + 1, self, dir)

    def __lt__(self, other: 'Node') -> bool:
        return self.fs < other.fs
    
    def __repr__(self) -> str:
        return "\n".join([" ".join(map(str, row)) for row in self.state])

    def __eq__(self, other: 'Node') -> bool:
        return self.state == other.state

    def __hash__(self) -> int:
        return self._hash


def IDA_star(start_state: list, args) -> list:
    """
    IDA* algorithm to solve the 15-puzzle problem.
    Args:
        start_state (list): The initial state of the puzzle.
    Returns:
        list: The path from the start state to the goal state.
    """
    start_node = Node(start_state, 0)
    threshold = start_node.hs

    path = [start_node]
    while True:
        if args.debug:
            print(f"Current threshold: {threshold}")
        success, fs_min = DepthLimitedSearch(path, start_node, threshold)
        if success:
            return path
        elif fs_min == float('inf'):
            return None
        threshold = fs_min


    # Depth-limited search
    # while threshold < float('inf'):
    #     stack = [start_node]
    #     fs_min = float('inf')
    #     while stack:
    #         cur_node = stack.pop()
    #         if cur_node.state == Node.target:
    #             return backtrace(cur_node)
            
    #         if cur_node.fs > threshold:
    #             fs_min = min(fs_min, cur_node.fs)
    #             continue

    #         for neighbor in cur_node.neighbors():
    #             stack.append(neighbor)

    #     threshold = fs_min
    
    return None

def DepthLimitedSearch(path: list, node: Node, threshold: int) -> tuple:
    """
    Perform a depth-limited search to find the goal state.
    Args:
        node (Node): The current node.
        threshold (int): The threshold for the search.
    Returns:
        tuple: A tuple containing the result and the minimum fs value.
    """
    if node.state == Node.target:
        return True, node.gs

    if node.fs > threshold:
        return False, node.fs

    fs_min = float('inf')
    for neighbor in node.neighbors():
        if neighbor in path:
            continue
        path.append(neighbor)
        found, fs = DepthLimitedSearch(path, neighbor, threshold)
        if found:
            return True, fs
        if fs < fs_min:
            fs_min = fs
        path.pop()

    return False, fs_min

def backtrace(cur):
    total_path = []
    while cur is not None:
        total_path.append(cur)
        cur = cur.parent
    return total_path[::-1]

def run(test_id: int, args) -> None:
    print(f"Running test {test_id}...")
    start_time = time.time()
    path = IDA_star(start_state[test_id], args)
    end_time = time.time()

    save_path = Path(__file__).resolve().parent.joinpath(f'out_IDA_{test_id}.txt')
    with open(save_path, "w") as f:
        dir = ["Right", "Down", "Left", "Up", "Start"]
        f.write(f"Time taken: {end_time - start_time:.2f} seconds\n")
        dir_brief = ["R", "D", "L", "U", "S"]
        path_str = ''
        for i, node in enumerate(path):
            path_str += dir_brief[node.direction]
        f.write(f"Path: {path_str}\n")
        f.write(f"Path length: {len(path) - 1}\n")
        for i, node in enumerate(path):
            f.write(f"Step {i}, {dir[node.direction]}:\n")
            f.write(repr(node))
            f.write("\n\n")

    print(f"\nTest {test_id} completed.")
    print(f"Number of steps: {len(path) - 1}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Output saved to:", save_path)

if __name__ == "__main__":
    parser = parse.ArgumentParser()
    parser.add_argument("debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
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
        run(test_id, args)
    # run(5)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
