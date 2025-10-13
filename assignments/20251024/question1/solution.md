# Question 1: Monte Carlo Tree Search (MCTS) 蒙特卡洛树搜索

## Part 1 (5 points): UCB Calculation and Path Selection | UCB计算与路径选择

### UCB Formula | UCB公式
```
UCB(node) = Q/N + C × sqrt(ln(N_parent) / N)
```

Where | 其中:
- Q = Estimated value of the node | 节点的估计值
- N = Number of visits to the node | 节点的访问次数
- N_parent = Number of visits to the parent node | 父节点的访问次数
- C = Exploration parameter (C=1 in this problem) | 探索参数（本题C=1）

### Step 1: Calculate UCB for Second Layer | 第一步：计算第二层的UCB

From root node (29/7), calculate UCB for three children:

从根节点(29/7)出发，计算三个子节点的UCB：

#### Left Node (1/1) | 左节点 (1/1)
- Exploitation term | 利用项: Q/N = 1/1 = 1
- Exploration term | 探索项: C × sqrt(ln(7)/1) = 1 × sqrt(1.946) ≈ 1.395
- **UCB = 1 + 1.395 = 2.395**

#### Middle Node (21/4) | 中节点 (21/4)
- Exploitation term | 利用项: Q/N = 21/4 = 5.25
- Exploration term | 探索项: C × sqrt(ln(7)/4) = 1 × sqrt(0.486) ≈ 0.697
- **UCB = 5.25 + 0.697 = 5.947** ✓ (Maximum | 最大值)

#### Right Node (7/2) | 右节点 (7/2)
- Exploitation term | 利用项: Q/N = 7/2 = 3.5
- Exploration term | 探索项: C × sqrt(ln(7)/2) = 1 × sqrt(0.973) ≈ 0.986
- **UCB = 3.5 + 0.986 = 4.486**

**Selection: Middle node (21/4)** | **选择：中节点 (21/4)**

---

### Step 2: Calculate UCB for Third Layer | 第二步：计算第三层的UCB

From middle node (21/4), calculate UCB for three children:

从中节点(21/4)出发，计算三个子节点的UCB：

#### Node (6/1) | 节点 (6/1)
- UCB = 6/1 + sqrt(ln(4)/1) = 6 + 1.177 = **7.177**

#### Node (2/1) | 节点 (2/1)
- UCB = 2/1 + sqrt(ln(4)/1) = 2 + 1.177 = **3.177**

#### Node (7/1) | 节点 (7/1)
- UCB = 7/1 + sqrt(ln(4)/1) = 7 + 1.177 = **8.177** ✓ (Maximum | 最大值)

**Selection: Node (7/1)** | **选择：节点 (7/1)**

---

### Final Answer | 最终答案

**Selected Path: Root → 21/4 → 7/1**

**选择的路径：根节点 → 21/4 → 7/1**

---

## Part 2 (5 points): MCTS Four Procedures | MCTS四步骤

### The Four Steps of MCTS | MCTS的四个步骤

1. **Selection (选择)**: Use UCB to traverse from root to a leaf node | 使用UCB从根节点遍历到叶节点
2. **Expansion (扩展)**: Add child nodes to the selected leaf | 为选中的叶节点添加子节点
3. **Rollout/Simulation (模拟)**: Simulate to terminal state and get utility | 模拟到终止状态并获得效用值
4. **Backpropagation (回传)**: Update statistics along the path | 沿路径更新统计信息

---

### Execution Process | 执行过程

#### Step 1: Selection | 选择
- Already completed in Part 1 | 已在第一部分完成
- Selected path: Root → 21/4 → 7/1 | 选择路径：根节点 → 21/4 → 7/1

#### Step 2: Expansion | 扩展
- Node 7/1 is a leaf node with utility value 7 | 节点 7/1 是叶节点，效用值为7
- Since it's already a terminal node, no expansion needed | 因为已是终止节点，无需扩展
- (Alternatively, if we need to expand, we would add its children as unexpanded nodes)
- （或者，如果需要扩展，我们会将其子节点添加为未展开节点）

#### Step 3: Rollout | 模拟
- The utility value obtained = 7 | 获得的效用值 = 7

#### Step 4: Backpropagation | 回传
Update nodes along the path (adding the rollout value and incrementing visit count):

沿路径更新节点（添加模拟值并增加访问次数）：

1. **Node (7/1)**:
   - New value: (7 + 7) / (1 + 1) = 14/2
   - **Updated: 14/2 (or 7/2 showing average)**

2. **Node (21/4)**:
   - New value: (21 + 7) / (4 + 1) = 28/5
   - **Updated: 28/5**

3. **Root Node (29/7)**:
   - New value: (29 + 7) / (7 + 1) = 36/8
   - **Updated: 36/8**

---

### Visualization of Updated Tree | 更新后的树的可视化

```
                    36/8
                   /  |  \
                1/1  28/5  7/2
               /|\   /|\   /|\
              ? ? ? 6/1 2/1 14/2  3/1 ? ?
              1 9 5  6   2   7     3  4 8
```

Note: The node (7/1) has been updated to (14/2) after backpropagation.

注意：节点(7/1)在回传后更新为(14/2)。

---

## Part 3 (10 points): Multiple Iterations and Algorithm Improvement | 多次迭代与算法改进

### Question | 问题
Can the leaf node with the largest utility (i.e., 9) be returned after more iterations? If not, suggest modifications for higher efficiency.

经过更多迭代后，能否返回具有最大效用值的叶节点（即9）？如果不能，请提出改进建议以提高效率。

**Can MCTS find the maximum utility 9? MCTS能找到最大效用值9吗？**

**Key Findings | 经过sulotion.py暴力1搜索后:**
1. Minimum C threshold: C ≥ 6.0
   - C < 6.0: Failed to find optimal leaf even with 100+ iterations
   - C = 6.0: Found at iteration 33
   - C ≥ 15.0: Found in just 3 iterations (fastest)
2. Complete Results Table:
   - C = 6.0 → n = 3 iterations
   - C = 7.0 → n = 15 iterations
   - C = 8.0 → n = 14 iterations
   - C = 9.0 → n = 10 iterations
   - C = 10.0 → n = 8 iterations
   - C = 15.0+ → n = 3 iterations (optimal)
3. Why This Happens:
   - Node A (parent of A2) starts with low value (1.0) and only 1 visit
   - After expanding A1 (utility=1), A's average stays low
   - Nodes B (5.25 avg) and C (3.5 avg) look more promising
   - The UCB formula naturally favors higher-value branches
   - Higher C values increase the exploration bonus, forcing the algorithm to revisit "unpromising" branches like A
   - This eventually leads to discovering A2 with its high utility of 9
4. Visualizations Generated:
   - All successful combinations have been visualized in assignments/20251024/pics/
   - The visualization shows 4 stages: Initial st

**Best approach | 最佳方法**:
1. Increase C to 1.5-2.0 | 将C增加到1.5-2.0
2. Run at least 50-100 iterations | 至少运行50-100次迭代
3. Consider implementing ε-greedy for guaranteed exploration
4. 考虑实现ε-贪心以保证探索