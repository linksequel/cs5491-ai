# 从零掌握MCTS

## UCB Formula | UCB公式
$UCB(node) = Q/N + C × sqrt(ln(N_parent) / N)$

Where | 其中:
- Q = Estimated value of the node | 节点的估计值
- N = Number of visits to the node | 节点的访问次数
- N_parent = Number of visits to the parent node | 父节点的访问次数
- C = Exploration parameter (C=1 in this problem) | 探索参数（本题C=1）

## 1. 面对题目提出问题
### Q: 叶节点的效用什么意思？
叶节点的 “效用（utility）” 是一个核心概念，它主要用于衡量叶节点所代表的状态或结果的 “好坏程度”“价值高低”。用来量化这个终局对当前决策方的价值
### Q: 非叶子节点上的数值什么意思？如21/4
在蒙特卡洛树搜索（MCTS）的这个例子中，非叶子节点上的“\( 21/4 \)”这类形式的数值，通常表示的是“**该节点的累计收益值 / 该节点被访问的次数**”。
- **分子（累计收益值）**：它是从该节点出发，经过模拟（比如“roll out”阶段的随机推演等过程）后，所有后续相关叶节点效用值的累计总和。例如，当从这个非叶子节点展开并模拟对局或决策流程，最终到达不同的叶节点，把这些叶节点的效用值加起来，就得到了该非叶子节点的累计收益。
- **分母（访问次数）**：代表这个非叶子节点在MCTS的迭代过程中，被选中并进行后续处理（如选择、扩展等步骤）的次数。
- 通过“累计收益值 / 访问次数”，可以得到该节点的平均收益，这是MCTS中用于评估节点“好坏”、指导后续选择（比如结合UCB公式选择更有潜力的节点）的重要依据。
- 21/4 可能的一个组合是 6+7+6+2

## 2. 探索/利用 思想
**key problem/model**: [多臂老虎机问题](https://www.bilibili.com/video/BV1za411a7KA/?spm_id_from=333.337.search-card.all.click)

可以简单理解为：假设有一台多臂老虎机，有k个拉杆，每个拉杆被拉动后获得奖励的概率和奖励值都不同，玩家在有限的尝试次数内，如何选择拉杆策略，以最大化获得的总奖励 。

  Summary of Results

  Question: Can the leaf node with the largest utility (utility=9, which is node A2) be found by increasing MCTS iterations?

  Answer: Yes! The optimal leaf A2 (utility=9) can be found, but it requires a sufficiently high exploration parameter C.

## Key Findings
1. Minimum C threshold: C ≥ 6.0
   - C < 6.0: Failed to find optimal leaf even with 100+ iterations
   - C = 6.0: Found at iteration 33
   - C ≥ 15.0: Found in just 3 iterations (fastest)
2. Complete Results Table:
   - C = 6.0 → n = 33 iterations
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
   - The visualization shows 4 stages: Initial state, Mid-iteration, When A2 is found, Final state