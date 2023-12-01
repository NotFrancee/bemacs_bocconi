class Node:
    def __init__(self, key, parent=None, left=None, right=None):
        self.key = key
        self.parent = parent
        self.left = left
        self.right = right

    def __repr__(self):
        s = f"Node({self.key}"
        if self.parent is not None:
            s += f", parent={self.parent.key}"
        if self.left is not None:
            s += f", left={self.left.key}"
        if self.right is not None:
            s += f", right={self.right.key}"
        s += ")"

        return s


class BST:
    def __init__(self, root=None):
        self.root = root
        self.size = 0

    def __repr__(self) -> str:
        s = f"BST(size={self.size}, root={self.root})"
        return s

    def insert(self, node):
        x = self.root
        p = None

        while x is not None:
            p = x
            if x.key > node.key:
                x = x.left
            elif x.key < node.key:
                x = x.right

            else:
                raise Exception("key already exists")

        # now x is none so we found our spot
        # we are sure node is going to be added to the tree
        self.size += 1

        if p is None:
            self.root = node
        elif p.key > node.key:
            node.parent = p.left = node
            # attach node on the left of p
        elif p.key < node.key:
            # attach node on the right of p
            node.parent = p.right = node

        return node


def build_BST(ls: list):
    # check that the elements are unique
    assert len(set(ls)) == len(ls)

    nodes = [Node(key) for key in ls]
    bst = BST()

    for node in nodes:
        bst.insert(node)

    return bst


print(build_BST([1, 11, 2, 3, 4, 65, 12, 8, 14]))
