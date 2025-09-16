### Week2_Tutorial.ipynb 核心思路与算法说明

本说明文档概述 `week2/Week2_Tutorial.ipynb` 的问题背景、图像到图模型的构建流程，以及使用 DFS/BFS 求解四色地图着色问题的核心算法与可改进方向。

### 1. 问题背景与形式化
- **任务**: 用不超过四种颜色给一个平面地图着色，要求任意相邻区域颜色不同（四色定理保证可行）。
- **状态（State）**: 已上色与未上色区域的颜色分配向量（长度为区域数）。
- **动作（Action）**: 为下一个未上色区域选择一种颜色（最多四种）。
- **转移（Transition）**: 将该区域标记为所选颜色。
- **约束/目标测试（Goal Test）**: 
  - 所有区域均已着色；
  - 对于任意一条邻接边，两端区域颜色不同。
- **代价（Cost）**: 计数为赋值次数（示例实现未优化代价，仅作比较）。

### 2. 图像预处理（由地图到像素语义）
- 读取输入图像 `iran.jpg`，限制最大尺寸 `MAXIMUM_IMAGE_WIDTH/HEIGHT`，过大则退出。
- 通过阈值函数 `apply_threshold()` 将背景（过暗或过亮）像素设为白色并标记为 `BACKGROUND_MARK`：
  - 利用 `IMPORTANT_COLOR_LOW_THRESHOLD` 与 `IMPORTANT_COLOR_HIGH_THRESHOLD` 对 RGB 和值进行筛分；
- 去噪与增强：
  - `cv2.medianBlur(image, 3)` 中值滤波；
  - `cv2.filter2D(image, -1, SHARPEN_KERNEL)` 锐化；
  - 每步后再次阈值化，巩固背景与前景的区分。

### 3. 区域检测与标注（像素连通到区域）
- 使用二维 `mark` 矩阵记录每个像素的区域标记，初始为 `NOT_MARKED`。
- 基于像素相似与4邻域连通，使用 `get_region_area()` 进行区域泛洪（队列式 BFS）：
  - 颜色接近通过 `same_pixel_colors()` 判定（RGB 差值和阈 3×`MAXIMUM_NEIGHBOR_PIXEL_COLOR_DIFFERENCE`）。
- 对每个新发现的连通块：
  - 若面积大于 `MINIMUM_REGION_AREA_RATIO * total_area`，则保留为有效区域并创建 `Node`；
  - 否则回滚为未标记（视为背景噪声）。
- 构建辅助索引：
  - `regions[region_id]`: 区域内像素列表；
  - `regions_border[region_id]`: 区域边界像素列表（`is_on_border()` 判断）。

### 4. 构图：区域图与邻接判定
- 结点结构 `Node{id, x, y, adj}`，`x,y` 取区域内代表点。
- 邻接判定 `are_adjacent(node1, node2)`：
  - 在两区域边界像素中找最近点对（欧氏距离平方最小）；
  - 快速拒绝：若最近距离平方超过 `MINIMUM_BORDER_WIDTH_RATIO * (W^2 + H^2)`，判非邻接；
  - 细查：在两最近边界点间做线性采样，若路径上出现第三方区域像素，则判非邻接；否则为邻接；
- 双向加边 `add_graph_edges()` 完成区域邻接图构建。

### 5. 背景处理与可视化准备
- `whiten_background()` 将背景与未标注像素统一涂白，以便后续着色可视化清晰。
- `show_image()` 使用 PIL 展示最终结果；搜索过程中使用 OpenCV `imshow` 动态显示着色进度。

### 6. 搜索建模与约束检查
- 树结点 `tNode{id, path}`：
  - `path` 为长度等于区域数的颜色数组，初始化为 `NO_COLOR`；
  - 深度 `id` 表示已决定到的最后一个区域索引。
- 约束检查 `check(id, path)`：
  - 对已决定的区域，沿已建图的邻接表 `nodes[i].adj` 检查冲突（相邻同色即失败）。
- 着色动作 `change_region_color(node, pixel_color)`：
  - 将对应区域内所有像素赋为指定 RGB 颜色，供可视化使用。

### 7. 两种无信息搜索：DFS 与 BFS
- 共同点：
  - 从给定起始颜色（如 `DFS(0)` / `BFS(0)`）开始；
  - 对当前部分赋值 `path`，尝试为下一区域依次选择 4 种颜色；
  - 仅当 `check` 通过时才扩展新结点；
  - 每次扩展后进行可视化刷新；
  - 当所有区域均赋值（`len(visited) == len(nodes)`）即返回解。
- 差异点：
  - DFS 使用列表作为栈（LIFO）进行深度优先；
  - BFS 使用 `queue.Queue` 作为队列（FIFO）进行广度优先；
- 实测示例中，BFS 随区域数增长呈指数级膨胀，明显慢于 DFS。

### 8. 关键参数（可调）
- `IMPORTANT_COLOR_LOW_THRESHOLD / HIGH_THRESHOLD`：前景与背景的颜色阈值；
- `MAXIMUM_NEIGHBOR_PIXEL_COLOR_DIFFERENCE`：像素相似阈，用于区域泛洪；
- `MINIMUM_REGION_AREA_RATIO`：最小有效区域占比；
- `MINIMUM_BORDER_WIDTH_RATIO`：邻接快速拒绝的距离阈；
- `MAXIMUM_IMAGE_WIDTH/HEIGHT`：图像尺寸限制；
- `COLORING_COLORS`：四色的 RGB 值；
- `SLEEP_TIME_IN_MILLISECONDS`：可视化刷新的等待时间。

### 9. 复杂度与性能观察
- 在无启发的条件下：
  - BFS 空间与时间消耗随分支因子与深度快速增长；
  - DFS 更快抵达某一可行解，但可能在差解路径上“走深”后回溯。
- 本实现将区域顺序固定为发现顺序、颜色顺序固定为先验四色，因此搜索效率主要受图结构与区域顺序影响。

### 10. 改进建议（CSP/启发式）
- 将问题表述为 CSP：变量=区域、域=颜色、约束=邻接不同色；
- 引入经典启发式：
  - 变量选择：MRV（最少剩余值）、度启发（最大冲突数优先）；
  - 值选择：LCV（最不约束邻居的颜色优先）；
  - 约束传播：前向检查、AC-3；
  - 回溯搜索：带上述启发式通常显著快于朴素 DFS/BFS；
- 图层面优化：
  - 预处理双连通分量/割点分解；
  - 依据图的团与着色上界做裁剪；
- 图像层面：
  - 更稳健的分割与边界提取（如形态学操作、边缘检测）。

### 11. 依赖与运行要点
- 依赖（示例环境）：
  - `opencv-python, numpy, matplotlib, Pillow, scipy, psutil` 等；
- 输入：`week2/iran.jpg`（确保尺寸不超过上限）。
- 运行：顺序执行 Notebook 单元；先预处理与构图，再运行 `DFS(0)` 或 `BFS(0)`，最后使用 `show_image()` 查看结果。

### 12. 小结
该 Notebook 将“图像地图”转化为“区域图”，再以无信息搜索（DFS/BFS）完成四色着色。核心在于：可靠的区域提取与邻接判定、正确的约束检查、以及搜索前沿结构的选择。进一步引入 CSP 启发与约束传播，可在更复杂的地图上显著提升效率。
