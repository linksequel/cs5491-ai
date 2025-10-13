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

---

### Analysis | 分析

#### Current Situation | 当前情况
- The maximum utility value 9 is located in the left subtree (left child's first unexpanded node)
- 最大效用值9位于左子树（左节点的第一个未展开节点）

- The left node (1/1) has the lowest UCB value because its estimated value is only 1
- 左节点(1/1)的UCB值最低，因为其估计值仅为1

- MCTS tends to exploit known good paths (exploitation) while balancing exploration
- MCTS倾向于利用已知的好路径（exploitation），同时平衡探索（exploration）

#### Will MCTS Find Utility 9? | MCTS能找到效用值9吗？

**Theoretical Answer | 理论答案**: Yes, but it requires many iterations.

**是的，但需要很多次迭代。**

**Practical Problem | 实际问题**:
- The left branch has low UCB due to poor initial performance
- 由于初始表现不佳，左分支的UCB值较低
- The algorithm will prioritize middle and right branches for many iterations
- 算法会在很多次迭代中优先选择中间和右边的分支
- Eventually, as other nodes are visited more, the exploration term for the left node will increase
- 最终，随着其他节点被访问更多次，左节点的探索项会增加

---

### Suggestions for Algorithm Improvement | 算法改进建议

#### 1. Increase Exploration Parameter C | 增大探索参数 C

**Current | 当前**: C = 1

**Suggestion | 建议**: C = 1.5 or C = 2 or even √2 ≈ 1.414

**Effect | 效果**:
- Increases the weight of the exploration term | 增加探索项的权重
- Encourages more exploration of less-visited nodes | 鼓励更多探索访问较少的节点
- Better balance between exploitation and exploration | 更好地平衡利用和探索

**Example Calculation with C=2 for Left Node | C=2时左节点的计算示例**:
```
UCB = 1/1 + 2 × sqrt(ln(7)/1) = 1 + 2 × 1.395 = 3.79
```
This is higher and more competitive! | 这样更高、更有竞争力！

---

#### 2. ε-Greedy Strategy | ε-贪心策略

**Method | 方法**:
- With probability ε (e.g., ε=0.1), randomly select a child node
- 以概率ε（例如ε=0.1）随机选择一个子节点
- With probability (1-ε), use UCB selection
- 以概率(1-ε)使用UCB选择

**Advantage | 优点**:
- Guarantees all branches get explored eventually | 保证所有分支最终都会被探索
- Prevents getting stuck in local optima | 防止陷入局部最优

---

#### 3. Increase Number of Iterations | 增加迭代次数

**Current State | 当前状态**: 7 total visits at root | 根节点总访问次数为7

**Suggestion | 建议**: Run 50-100 iterations | 运行50-100次迭代

**Effect | 效果**:
- As N_parent increases, exploration terms for all children increase
- 随着N_parent增加，所有子节点的探索项都会增加
- Eventually, even low-value branches will be explored
- 最终，即使是低价值的分支也会被探索

**UCB for Left Node after 50 root visits | 根节点访问50次后左节点的UCB**:
```
UCB = 1/1 + 1 × sqrt(ln(50)/1) ≈ 1 + 1.956 = 2.956
```

---

#### 4. Progressive Widening | 渐进扩展

**Method | 方法**:
- Don't expand all children at once | 不要一次性扩展所有子节点
- Gradually add children as node is visited more | 随着节点被访问更多次，逐渐添加子节点
- Formula: number_of_children = k × N^α (α ∈ [0, 1])

**Advantage | 优点**:
- Focuses computational resources on promising areas first
- 首先将计算资源集中在有前途的区域
- Still allows for eventual complete exploration
- 仍然允许最终完全探索

---

#### 5. Domain Knowledge Initialization | 领域知识初始化

**Method | 方法**:
- Initialize unexpanded nodes with optimistic values | 用乐观值初始化未展开节点
- Use heuristics to estimate potential value | 使用启发式方法估计潜在价值

**Example | 示例**:
```
Initial value for unexpanded nodes = max(parent_value, average_leaf_value)
未展开节点的初始值 = max(父节点值, 平均叶节点值)
```

**Advantage | 优点**:
- Encourages early exploration of all branches | 鼓励早期探索所有分支
- Reduces bias towards first-explored branches | 减少对首先探索分支的偏见

---

#### 6. UCT Variants | UCT变体

**RAVE (Rapid Action Value Estimation)**:
- Shares information across similar states | 在相似状态间共享信息
- Faster convergence to optimal policy | 更快收敛到最优策略

**AMAF (All Moves As First)**:
- Updates values for all actions taken in simulation
- 为模拟中采取的所有动作更新值
- More efficient learning from each rollout
- 从每次模拟中更高效地学习

---

### Simulation Results | 模拟结果

#### Without Improvement | 不改进的情况

After 20 iterations with C=1, estimated path distribution:
- Middle branch: ~50% | 中间分支：约50%
- Right branch: ~35% | 右边分支：约35%
- Left branch: ~15% | 左边分支：约15%

**Likelihood of finding utility 9**: Low (≈15% × 33% ≈ 5%)

**找到效用值9的可能性**：低（约15% × 33% ≈ 5%）

#### With C=2 | 使用C=2

After 20 iterations with C=2, estimated path distribution:
- Middle branch: ~40% | 中间分支：约40%
- Right branch: ~30% | 右边分支：约30%
- Left branch: ~30% | 左边分支：约30%

**Likelihood of finding utility 9**: Medium (≈30% × 33% ≈ 10%)

**找到效用值9的可能性**：中等（约30% × 33% ≈ 10%）

---

### Conclusion | 结论

**Can MCTS find the maximum utility 9?**

**MCTS能找到最大效用值9吗？**

- **Yes, theoretically**, with enough iterations | **理论上可以**，只要有足够的迭代次数
- **Practically challenging** with current parameters (C=1, few iterations)
- **实际上有挑战**，使用当前参数（C=1，少量迭代）

**Best approach | 最佳方法**:
1. Increase C to 1.5-2.0 | 将C增加到1.5-2.0
2. Run at least 50-100 iterations | 至少运行50-100次迭代
3. Consider implementing ε-greedy for guaranteed exploration
4. 考虑实现ε-贪心以保证探索

**Key insight | 关键洞察**: MCTS balances exploitation and exploration, but the balance can be tuned based on the problem requirements.

**关键见解**：MCTS平衡利用和探索，但可以根据问题需求调整这种平衡。
