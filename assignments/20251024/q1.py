"""
MCTS (Monte Carlo Tree Search) Implementation with UCB Formula
Based on HW_1 Question 1

This implementation:
1. Initializes hyperparameters C and iteration count n
2. Outputs and visualizes the four steps of MCTS for each iteration:
   - Selection: Select the best child using UCB formula
   - Expansion: Expand an unexpanded child node
   - Rollout: Simulate to get a reward (use leaf utility values)
   - Backpropagation: Update values and visit counts along the path
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import math
import copy


class MCTSNode:
    """Node in the MCTS tree"""

    def __init__(self, name, parent=None, utility=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.utility = utility  # Leaf node utility value
        self.visits = 0
        self.value = 0.0
        self.is_expanded = False

    def is_leaf(self):
        """Check if node is a leaf node"""
        return self.utility is not None

    def is_fully_expanded(self):
        """Check if all children are expanded"""
        return self.is_expanded and len(self.children) > 0

    def add_child(self, child):
        """Add a child node"""
        self.children.append(child)

    def ucb_score(self, c, parent_visits):
        """Calculate UCB score"""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def __repr__(self):
        if self.visits == 0:
            return f"{self.name}(unexpanded)"
        return f"{self.name}({self.value:.1f}/{self.visits})"


class MCTS:
    """Monte Carlo Tree Search Algorithm"""

    def __init__(self, C=1.0, n_iterations=10, show_process=True):
        """
        Initialize MCTS

        Args:
            C: Exploration parameter for UCB formula
            n_iterations: Number of iterations to run
            show_process: Whether to visualize each iteration (True) or only the last one (False)
        """
        self.C = C
        self.n_iterations = n_iterations
        self.show_process = show_process
        self.root = None
        self.iteration_history = []

    def initialize_tree(self):
        """Initialize the tree based on HW_1 Question 1 figure"""
        # Create root node
        self.root = MCTSNode("Root")
        self.root.visits = 7
        self.root.value = 29.0
        self.root.is_expanded = True

        # Create first level
        node_1 = MCTSNode("A", parent=self.root)
        node_1.visits = 1
        node_1.value = 1.0
        node_1.is_expanded = False  # Children not expanded yet

        node_2 = MCTSNode("B", parent=self.root)
        node_2.visits = 4
        node_2.value = 21.0
        node_2.is_expanded = True

        node_3 = MCTSNode("C", parent=self.root)
        node_3.visits = 2
        node_3.value = 7.0
        node_3.is_expanded = True

        self.root.add_child(node_1)
        self.root.add_child(node_2)
        self.root.add_child(node_3)

        # Create second level (leaf nodes)
        # Under A (not expanded yet, but have potential children)
        leaf_1a = MCTSNode("A1", parent=node_1, utility=1)
        leaf_1b = MCTSNode("A2", parent=node_1, utility=9)
        leaf_1c = MCTSNode("A3", parent=node_1, utility=5)

        # Under B
        leaf_2a = MCTSNode("B1", parent=node_2, utility=6)
        leaf_2a.visits = 1
        leaf_2a.value = 6.0

        leaf_2b = MCTSNode("B2", parent=node_2, utility=2)
        leaf_2b.visits = 1
        leaf_2b.value = 2.0

        leaf_2c = MCTSNode("B3", parent=node_2, utility=7)
        leaf_2c.visits = 1
        leaf_2c.value = 7.0

        node_2.add_child(leaf_2a)
        node_2.add_child(leaf_2b)
        node_2.add_child(leaf_2c)

        # Under C
        leaf_3a = MCTSNode("C1", parent=node_3, utility=3)
        leaf_3a.visits = 1
        leaf_3a.value = 3.0

        leaf_3b = MCTSNode("C2", parent=node_3, utility=4)
        leaf_3c = MCTSNode("C3", parent=node_3, utility=8)

        node_3.add_child(leaf_3a)
        node_3.add_child(leaf_3b)
        node_3.add_child(leaf_3c)

        # Store unexpanded children for later expansion
        self._unexpanded_children = {
            'A': [leaf_1a, leaf_1b, leaf_1c],
            'B': [],
            'C': [leaf_3b, leaf_3c]
        }

    def select(self, node):
        """
        Selection phase: Select the best child using UCB formula
        """
        path = [node]

        while node.is_fully_expanded() and len(node.children) > 0:
            # Calculate UCB scores for all children
            best_score = -float('inf')
            best_child = None

            for child in node.children:
                score = child.ucb_score(self.C, node.visits)
                if score > best_score:
                    best_score = score
                    best_child = child

            node = best_child
            path.append(node)

        return node, path

    def expand(self, node):
        """
        Expansion phase: Expand an unexpanded child
        """
        # If node has unexpanded children, expand one
        if node.name in self._unexpanded_children and len(self._unexpanded_children[node.name]) > 0:
            child = self._unexpanded_children[node.name].pop(0)
            node.add_child(child)
            if len(self._unexpanded_children[node.name]) == 0:
                node.is_expanded = True
            return child

        # If all children are expanded but node was not marked
        if len(node.children) > 0:
            node.is_expanded = True
            # Return the first unvisited child if any
            for child in node.children:
                if child.visits == 0:
                    return child

        return node

    def rollout(self, node):
        """
        Rollout phase: Get the utility value
        For leaf nodes, return the utility value directly
        """
        if node.is_leaf():
            return node.utility

        # If not a leaf, return a random child's utility
        if len(node.children) > 0:
            child = np.random.choice(node.children)
            return self.rollout(child)

        return 0

    def backpropagate(self, path, reward):
        """
        Backpropagation phase: Update values and visit counts
        """
        for node in path:
            node.visits += 1
            node.value += reward

    def _create_tree_snapshot(self):
        """Create a deep copy snapshot of the tree state"""
        snapshot = {}
        snapshot['Root'] = {'visits': self.root.visits, 'value': self.root.value}

        # Track unexpanded children info
        unexpanded_info = {}

        for child in self.root.children:
            snapshot[child.name] = {'visits': child.visits, 'value': child.value, 'is_expanded': child.is_expanded}

            # Add all children (both expanded and unexpanded)
            for grandchild in child.children:
                snapshot[grandchild.name] = {
                    'visits': grandchild.visits,
                    'value': grandchild.value,
                    'utility': grandchild.utility
                }

            # Add unexpanded children info
            if child.name in self._unexpanded_children:
                for unexpanded_child in self._unexpanded_children[child.name]:
                    snapshot[unexpanded_child.name] = {
                        'visits': 0,
                        'value': 0,
                        'utility': unexpanded_child.utility,
                        'unexpanded': True
                    }
                    if child.name not in unexpanded_info:
                        unexpanded_info[child.name] = []
                    unexpanded_info[child.name].append(unexpanded_child.name)

        snapshot['_unexpanded_info'] = unexpanded_info
        return snapshot

    def run_iteration(self, iteration_num):
        """Run one iteration of MCTS"""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration_num + 1}")
        print(f"{'='*60}")

        # Save initial state (before backpropagation)
        initial_snapshot = self._create_tree_snapshot()

        # 1. Selection
        selected_node, path = self.select(self.root)
        print(f"\n1. SELECTION:")
        print(f"   Path: {' -> '.join([node.name for node in path])}")
        print(f"   Selected node: {selected_node}")

        # 2. Expansion
        expanded_node = self.expand(selected_node)
        if expanded_node != selected_node:
            path.append(expanded_node)
        print(f"\n2. EXPANSION:")
        print(f"   Expanded node: {expanded_node.name}")

        # 3. Rollout
        reward = self.rollout(expanded_node)
        print(f"\n3. ROLLOUT:")
        print(f"   Reward from {expanded_node.name}: {reward}")

        # 4. Backpropagation
        self.backpropagate(path, reward)
        print(f"\n4. BACKPROPAGATION:")
        print(f"   Updated path: {' -> '.join([str(node) for node in path])}")

        # Save final state (after backpropagation)
        final_snapshot = self._create_tree_snapshot()

        # Store iteration info for visualization
        selected_path_nodes = path[:len(path)-1] if expanded_node != selected_node else path
        iteration_info = {
            'iteration': iteration_num + 1,
            'selected_path': [node.name for node in selected_path_nodes],
            'expanded_node': expanded_node.name,
            'reward': reward,
            'updated_path': path,
            'initial_snapshot': initial_snapshot,
            'final_snapshot': final_snapshot
        }
        self.iteration_history.append(iteration_info)

        # Print current tree state
        self.print_tree()

    def print_tree(self):
        """Print the current state of the tree"""
        print(f"\nCurrent Tree State:")
        print(f"Root: {self.root}")
        for child in self.root.children:
            print(f"  {child}")
            for grandchild in child.children:
                print(f"    {grandchild}")

    def visualize_initial_state(self):
        """Visualize the initial state of the tree"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        fig.suptitle('MCTS Initial State', fontsize=16, fontweight='bold')

        # Get current snapshot
        snapshot = self._create_tree_snapshot()
        self._draw_tree(ax, snapshot=snapshot)
        ax.set_title('Initial Tree State Before Any Iteration', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('pics/mcts_initial_state.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Initial state visualization saved as pics/mcts_initial_state.png")

    def run(self):
        """Run MCTS for n iterations"""
        print(f"Starting MCTS with C={self.C}, n_iterations={self.n_iterations}")
        print(f"Initial tree state:")
        self.print_tree()

        # Visualize initial state
        print(f"\nVisualizing initial state...")
        self.visualize_initial_state()

        for i in range(self.n_iterations):
            self.run_iteration(i)

        # Final analysis
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        self.print_tree()

        # Find the best leaf
        best_leaf = None
        best_avg_value = -float('inf')

        for child in self.root.children:
            for grandchild in child.children:
                if grandchild.visits > 0:
                    avg_value = grandchild.value / grandchild.visits
                    if avg_value > best_avg_value:
                        best_avg_value = avg_value
                        best_leaf = grandchild

        if best_leaf:
            print(f"\nBest path found: Root -> {best_leaf.parent.name} -> {best_leaf.name}")
            print(f"Average value: {best_avg_value:.2f}")
            print(f"Utility: {best_leaf.utility}")

        # Check if we found the optimal leaf (utility = 9)
        optimal_found = False
        for child in self.root.children:
            for grandchild in child.children:
                if grandchild.utility == 9 and grandchild.visits > 0:
                    optimal_found = True
                    break

        print(f"\nOptimal leaf (utility=9) found: {optimal_found}")

        if not optimal_found:
            print("\nSuggestions for improvement:")
            print("1. Increase exploration parameter C to explore more unexpanded nodes")
            print("2. Increase number of iterations")
            print("3. Use progressive widening to better balance exploration vs exploitation")
            print("4. Add domain knowledge to guide selection towards promising branches")

    def visualize_iteration(self, iteration_num):
        """Visualize a specific iteration showing the 4 steps"""
        if iteration_num >= len(self.iteration_history):
            print(f"Iteration {iteration_num + 1} not available")
            return

        info = self.iteration_history[iteration_num]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'MCTS Iteration {info["iteration"]} - Four Steps', fontsize=16, fontweight='bold')

        # Step 1: Selection (use initial snapshot - before backprop)
        ax1 = axes[0, 0]
        self._draw_tree(ax1, snapshot=info['initial_snapshot'], highlight_path=info['selected_path'])
        ax1.set_title('Step 1: Selection\n(UCB formula selects best path)', fontsize=12, fontweight='bold')

        # Step 2: Expansion (use initial snapshot - before backprop)
        ax2 = axes[0, 1]
        self._draw_tree(ax2, snapshot=info['initial_snapshot'], highlight_node=info['expanded_node'])
        ax2.set_title(f'Step 2: Expansion\n(Expand node: {info["expanded_node"]})', fontsize=12, fontweight='bold')

        # Step 3: Rollout (use initial snapshot - before backprop)
        ax3 = axes[1, 0]
        self._draw_tree(ax3, snapshot=info['initial_snapshot'], highlight_node=info['expanded_node'], show_reward=info['reward'])
        ax3.set_title(f'Step 3: Rollout\n(Reward = {info["reward"]})', fontsize=12, fontweight='bold')

        # Step 4: Backpropagation (use final snapshot - after backprop)
        ax4 = axes[1, 1]
        path_names = [node.name for node in info['updated_path']]
        self._draw_tree(ax4, snapshot=info['final_snapshot'], highlight_path=path_names, backprop=True)
        ax4.set_title('Step 4: Backpropagation\n(Update values along path)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'pics/mcts_iteration_{info["iteration"]}.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved as mcts_iteration_{info['iteration']}.png")

    def _draw_tree(self, ax, snapshot=None, highlight_path=None, highlight_node=None, show_reward=None, backprop=False):
        """Draw the tree structure using a snapshot"""
        ax.clear()
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 8)
        ax.axis('off')

        # Use snapshot if provided, otherwise use current tree state
        if snapshot is None:
            snapshot = self._create_tree_snapshot()

        # Node positions
        positions = {
            'Root': (4.5, 7),
            'A': (1.5, 4.5),
            'B': (4.5, 4.5),
            'C': (7.5, 4.5),
            'A1': (0.5, 2), 'A2': (1.5, 2), 'A3': (2.5, 2),
            'B1': (3.5, 2), 'B2': (4.5, 2), 'B3': (5.5, 2),
            'C1': (6.5, 2), 'C2': (7.5, 2), 'C3': (8.5, 2)
        }

        # Draw edges (solid lines for expanded, dashed for unexpanded)
        edges = [
            ('Root', 'A'), ('Root', 'B'), ('Root', 'C'),
            ('B', 'B1'), ('B', 'B2'), ('B', 'B3'),
            ('C', 'C1')
        ]

        # Add edges for expanded nodes based on snapshot
        if 'A' in snapshot and snapshot['A'].get('is_expanded', False):
            edges.extend([('A', 'A1'), ('A', 'A2'), ('A', 'A3')])

        # Check if C has more children in snapshot
        if 'C2' in snapshot:
            edges.append(('C', 'C2'))
        if 'C3' in snapshot:
            edges.append(('C', 'C3'))

        # Get unexpanded edges info
        unexpanded_info = snapshot.get('_unexpanded_info', {})
        unexpanded_edges = []
        for parent, children_list in unexpanded_info.items():
            for child in children_list:
                unexpanded_edges.append((parent, child))

        # Draw solid edges (expanded nodes)
        for parent, child in edges:
            if parent in positions and child in positions:
                # Only draw if child exists in snapshot
                if child not in snapshot:
                    continue

                x1, y1 = positions[parent]
                x2, y2 = positions[child]

                # Highlight path
                if highlight_path and parent in highlight_path and child in highlight_path:
                    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, zorder=1)
                elif backprop and highlight_path and parent in highlight_path and child in highlight_path:
                    ax.plot([x1, x2], [y1, y2], 'g-', linewidth=3, zorder=1)
                else:
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, zorder=1)

        # Draw dashed edges (unexpanded nodes)
        for parent, child in unexpanded_edges:
            if parent in positions and child in positions:
                if child not in snapshot:
                    continue

                x1, y1 = positions[parent]
                x2, y2 = positions[child]

                # Draw dashed line for unexpanded children
                ax.plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.5, zorder=1)

        # Draw nodes using snapshot data
        for name, pos in positions.items():
            if name in snapshot:
                node_data = snapshot[name]
                x, y = pos

                visits = node_data.get('visits', 0)
                value = node_data.get('value', 0)
                utility = node_data.get('utility', None)

                # Node color
                if highlight_node and name == highlight_node:
                    color = 'yellow'
                    edgecolor = 'red'
                    linewidth = 3
                elif highlight_path and name in highlight_path:
                    color = 'lightcoral' if not backprop else 'lightgreen'
                    edgecolor = 'red' if not backprop else 'green'
                    linewidth = 2
                elif visits == 0:
                    color = 'lightgray'
                    edgecolor = 'gray'
                    linewidth = 1
                else:
                    color = 'lightblue'
                    edgecolor = 'black'
                    linewidth = 1

                # Draw circle
                circle = plt.Circle((x, y), 0.4, color=color, ec=edgecolor, linewidth=linewidth, zorder=2)
                ax.add_patch(circle)

                # Draw text
                if visits > 0:
                    text = f"{name}\n{value:.0f}/{visits}"
                else:
                    text = f"{name}"

                ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

                # Show utility for leaf nodes
                if utility is not None:
                    ax.text(x, y - 0.7, f"u={utility}", ha='center', va='top', fontsize=8, color='blue', zorder=3)

                # Show reward
                if show_reward and name == highlight_node:
                    ax.text(x + 1, y, f"reward={show_reward}", ha='left', va='center',
                           fontsize=10, color='red', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), zorder=4)


def main():
    """Main function to run MCTS"""
    # Initialize parameters
    C = 1.5  # Exploration parameter
    n_iterations = 1  # Number of iterations
    show_process = False  # Show each iteration (True) or only the last one (False)

    # Create MCTS instance
    mcts = MCTS(C=C, n_iterations=n_iterations, show_process=show_process)

    # Initialize tree with data from HW_1
    mcts.initialize_tree()

    # Run MCTS
    mcts.run()

    # Visualize iterations based on show_process flag
    print(f"\n{'='*60}")
    if mcts.show_process:
        print("Generating visualizations for each iteration...")
    else:
        print("Generating visualization for the last iteration only...")
    print(f"{'='*60}")

    if mcts.show_process:
        # Show all iterations
        for i in range(min(n_iterations, len(mcts.iteration_history))):
            mcts.visualize_iteration(i)
    else:
        # Show only the last iteration
        if len(mcts.iteration_history) > 0:
            last_iteration = len(mcts.iteration_history) - 1
            mcts.visualize_iteration(last_iteration)

    print("\nDone! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
