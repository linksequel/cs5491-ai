# DFS example for the tree in the provided image
# Tree structure (visual):
#        A
#       / \
#      B   C
#     / \ / \
#    D  E F  G
# E is the goal node (highlighted in picture)

from typing import Dict, List, Set

# Build the tree using adjacency lists (ordered left-to-right as in image)
TREE: Dict[str, List[str]] = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": [],
    "E": [],
    "F": [],
    "G": [],
}

GOAL = "G"


def dfs_recursive(node: str, tree: Dict[str, List[str]], visited: Set[str], order: List[str]) -> bool:
    """Perform recursive DFS. Append nodes to order list in visit order.
    Return True if goal found (to allow early exit), otherwise False."""
    visited.add(node)
    order.append(node)
    if node == GOAL:
        return True
    for child in tree.get(node, []):
        if child not in visited:
            found = dfs_recursive(child, tree, visited, order)
            if found:
                return True
    return False


def dfs_iterative(start: str, tree: Dict[str, List[str]]) -> List[str]:
    """Perform iterative DFS using an explicit stack. Return visit order.
    To mimic recursive left-to-right traversal, push children onto the stack in reverse order."""
    visited: Set[str] = set()
    order: List[str] = []
    stack: List[str] = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if node == GOAL:
            break
        # push children in reverse so left child is processed first
        children = tree.get(node, [])
        for child in reversed(children):
            if child not in visited:
                stack.append(child)
    return order


if __name__ == "__main__":
    # Recursive DFS
    visited_set: Set[str] = set()
    recursive_order: List[str] = []
    dfs_recursive("A", TREE, visited_set, recursive_order)
    print("Recursive DFS order:", " -> ".join(recursive_order))
    print("Goal found:" , GOAL in visited_set)

    # Iterative DFS
    iterative_order = dfs_iterative("A", TREE)
    print("Iterative DFS order:", " -> ".join(iterative_order))
    print("Goal found:", GOAL in iterative_order)
