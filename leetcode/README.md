# :mag: Project Guide

This warehouse is used to store the algorithm problems obtained from practicing Leetcode. It retains the more classic problems, including sorting, searching, and various algorithm challenges. For more information, visit the [LeetCode official website](https://leetcode.cn/problems).  

## :book: Algorithm Introduction

___1.回溯法___: 采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：找到一个可能存在的正确的答案；在尝试了所有可能的分步方法后宣告该问题没有答案。  
回溯算法是一种通过递归来解决问题的算法，它尝试分步构建解决方案，并在发现当前路径无法达到目标时撤销最近的选择（即“回溯”），并尝试其他可能的选择。回溯算法常用于解决组合问题、排列问题、子集问题、搜索问题等。 回溯算法的步骤如下：(1).定义问题：首先，你需要定义问题的要求和限制。问题可以是一个组合问题、排列问题、子集问题、搜索问题等。(2).确定搜索空间：确定问题的搜索空间，即所有可能的选择。(3).定义递归关系：定义递归关系，即如何从当前状态转移到下一个状态。(4).确定终止条件：确定递归的终止条件，即当搜索空间exhausted时，算法结束。

```python
def backtrack(path, options):
    # 结束条件：当满足某个条件时结束递归
    if is_a_solution(path):
        record_solution(path)
        return
    # 遍历所有可能的选择
    for choice in options:
        if is_valid(choice, path):  # 检查选择是否有效
            select(choice, path)     # 做出选择
            backtrack(path, new_options)  # 递归调用，继续探索
            undo(choice, path)       # 撤销选择，进行回溯
```

___2.深度优先搜索法___: Depth-First-Search，DFS是一种用于遍历或搜索树或图的算法。这个算法会尽可能深的搜索树的分支。当结点v的所在边都己被探寻过，搜索将回溯到发现结点v的那条边的起始结点。这一过程一直进行到已发现从源结点可达的所有结点为止。如果还存在未被发现的结点，则选择其中一个作为源结点并重复以上过程，整个进程反复进行直到所有结点都被访问为止。

```python
def dfs(node, visited, graph):
    if node in visited:
        return
    visited.add(node)               # 标记为已访问
    process(node)                   # 处理当前节点（如打印、记录等）
    for neighbor in graph[node]:    # 遍历所有邻接节点
        dfs(neighbor, visited, graph)
```

___3.广度优先搜索法___: Breadth-First-Search，广度优先搜索（BFS）是一种用于遍历或搜索树或图的算法。它从根节点（或起始节点）开始，逐层向外扩展，先访问所有距离为 1 的节点，再访问距离为 2 的节点，依此类推。BFS 保证在无权图中找到最短路径（按边数计算）。核心思想：优先向“广度”（横向）方向探索；使用队列（FIFO） 管理待访问节点；按“层”顺序访问，具有天然的最短路径性质（无权图）。

```python
from collections import deque
def bfs(start, graph):
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()      # 从队首取出
        process(node)               # 处理当前节点
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```
