import numpy as np
import time
from pathlib import Path
from queue import PriorityQueue

class Node:
    target = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
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
            gs: int = int(1e8),
            direction: int = -1):
        self.state = np.array(state)
        self.parent = parent
        self.gs = gs
        self.direction = direction
        self.hs = self.hScore()
        self.fs = self.gs + self.hs
        self._hash = hash(self.state.tobytes())

    def hScore(self) -> int:
        # return np.sum(self.state != self.target)
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

    # def fScore(self) -> int:
    #     return self.gs + self.hs
    
    # def gScore(self) -> int:
    #     return self.gs
    
    def neighbors(self):
        """
        Generate all possible states by moving the blank space
        Returns:
            
        """
        x, y = np.where(self.state == 0)
        x, y = x[0], y[0]
        for dir, (dx, dy) in enumerate(self.directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4:
                new_state = self.state.copy()
                new_state[x, y], new_state[nx, ny] = new_state[nx, ny], new_state[x, y]
                yield Node(new_state, self, self.gs + 1, dir)

    def __lt__(self, other: 'Node') -> bool:
        return self.fs < other.fs
    
    def __repr__(self) -> str:
        return str(self.state)
    
    def __eq__(self, other: 'Node') -> bool:
        return np.array_equal(self.state, other.state)

    def __hash__(self) -> int:
        return self._hash
    

def a_star(start: list) -> list:
    """
    Implements the A* algorithm to find the shortest path to the target state.
    Args:
        start (list of list): The initial state of the puzzle.
    Returns:
        list: The sequence of states leading to the solution.
    """
    open_set = PriorityQueue()
    start = Node(start, None, 0)
    open_set.put(start)
    close_set = set()
    gScore = {start: 0}
    came_from = {}
    while not open_set.empty():
        cur = open_set.get()
        # if cur.gs > gScore[cur]:
            # continue
        if cur in close_set:
            continue
        close_set.add(cur)
        if np.array_equal(cur.state, Node.target):
            return reconstruct_path(came_from, cur)
        for neighbor in cur.neighbors():
            if neighbor not in gScore or neighbor.gs < gScore[neighbor]:
                came_from[neighbor] = cur
                gScore[neighbor] = neighbor.gs
                # if neighbor in open_set.queue:
                    # open_set.queue.remove(neighbor)
                open_set.put(neighbor)
    return []

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


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

    test_id = 0
    path = a_star(start_state[test_id])

    # for i, state in enumerate(path):
    #     print(f"Step {i}:")
    #     print(state)
    #     print()
    
    end_time = time.time()
    # save to file
    save_path = Path(__file__).resolve().parent
    with open(save_path.joinpath(f'output_{test_id}.txt'), "w") as f:
        dir = ["Right", "Down", "Left", "Up", "Start"]
        f.write(f"Time taken: {end_time - start_time:.2f} seconds\n\n")
        for i, node in enumerate(path):
            f.write(f"Step {i}, {dir[node.direction]}:\n")
            f.write(repr(node))
            f.write("\n\n")

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Output saved to:", save_path.joinpath(f'output_{test_id}.txt'))
