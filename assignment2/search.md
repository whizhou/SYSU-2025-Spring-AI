# 23336020 周子健 Week3 实验报告

## 一、实验题目

+ 使用启发式搜索算法 A* 和 IDA* 解决15-Puzzle问题
+ 利用遗传算法求解TSP问题

## 二、实验内容

### 1. 算法原理

+ **A star算法**：

  定义起点 $s$，终点 $t$，从起点（初始状态）开始的距离函数 $g(x)$，到终点（最终状态）的距离函数 $h(x), h^*(x)$，其中 $h(x)$ 为启发式函数，$h^*(x)$ 为实际的距离，以及每个点的估价函数 $f(x)=g(x)+h(x)$

  A* 算法每次从优先队列中取出一个 $f$ 最小的元素，然后更新相邻状态。

  对于启发式函数 $h$：

  + 如果 $h \leq h^*$，则 A* 算法能找到最优解
  + 上述条件下，如果 $h$ 满足三角不等式，则 A* 不会将重复结点加入队列
  + 当 $h=0$ 时，A* 算法变为 Dijkstra；当 $h=0$ 且边权为 1 时变为 BFS

+ **IDA star 算法**：

  IDA* 为采用了迭代加深算法的 A* 算法，相对于 A* 算法，优缺点如下：

  1. 不需要判重，不需要排序，利于深度剪枝。
  2. 空间需求减少：每个深度下实际上是一个深度优先搜索，不过深度有限制，使用 DFS 可以减小空间消耗。
  3. 重复搜索：即使前后两次搜索相差微小，回溯过程中每次深度变大都要再次从头搜索。

  伪代码：

  ```c
  Procedure IDA_STAR(StartState)
  Begin
    PathLimit := H(StartState) - 1;
    Succes := False;
    Repeat
      inc(PathLimit);
      StartState.g = 0;
      Push(OpenStack, StartState);
      Repeat
        CurrentState := Pop(OpenStack);
        If Solution(CurrentState) then
          Success = True
        Elseif PathLimit >= CurrentState.g + H(CurrentState) then
          For each Child(CurrentState) do
            Push(OpenStack, Child(CurrentState));
      until Success or empty(OpenStack);
    until Success or ResourceLimtsReached;
  end;
  ```

+ **遗传算法**：

  1. 选择初始生命种群

  2. 评价种群中的个体适应度
  3. 以比例原则选择产生下一个种群（轮盘法，竞争法）
  4. 交叉和变异
  5. 重复2-4步直到停止循环的条件满足

  利用遗传算法解决TSP问题伪代码
  
  ![image-20250403172907149](./search.assets/image-20250403172907149.png)

### 2. 关键代码展示

#### 启发式搜索

+ 为了简化与统一算法代码，首先定义 `Node` 类，实现状态的创建、扩展、格式化输出等：

  ```python
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
              row = [self.state[i][j] for j in range(4) if self.state[i][j] != 0]
              for j in range(len(row) - 1):
                  for k in range(j + 1, len(row)):
                      if (row[j] // 4 == row[k] // 4) and (row[j] % 4 > row[k] % 4):
                          distance += 2
              col = [self.state[j][i] for j in range(4) if self.state[j][i] != 0]
              for j in range(len(col) - 1):
                  for k in range(j + 1, len(col)):
                      if (col[j] % 4 == col[k] % 4) and (col[j] // 4 > col[k] // 4):
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
  ```

  

### 3. 创新点 & 优化

## 三、实验结果及分析

### 1. 实验结果展示实力

### 2. 评测指标展示及分析

## 四、参考资料

