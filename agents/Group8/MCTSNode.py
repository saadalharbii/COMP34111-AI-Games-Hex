# agents/Group8/MCTSNode.py

class MCTSNode:
    """
    A node in the MCTS tree.
    Supports both simple UCT and PUCT (for NN-guided MCTS).
    """

    def __init__(self, parent=None, prior_p=1.0):
        # Tree structure
        self.parent = parent
        self.children = {}  # (x,y) -> child node

        # MCTS statistics
        self.N = 0      # visit count
        self.W = 0.0    # total value
        self.Q = 0.0    # mean value

        # Prior from neural network (PUCT)
        self.P = prior_p

        # Moves not yet expanded
        self.unexpanded_moves = None  # assigned during expansion

        # Optional cached NN value for the state
        self.value_from_nn = None


    def is_expanded(self):
        """
        A node is expanded once we have recorded its legal moves.
        """
        return self.unexpanded_moves is not None


    def expand(self, legal_moves, move_priors=None):
        """
        Initialize this node's unexpanded moves list.

        legal_moves: list of (x, y)
        move_priors: optional dict {(x,y): P}
        """
        if move_priors is None:
            # uniform priors if NN is not integrated yet
            self.unexpanded_moves = list(legal_moves)
        else:
            # we might attach priors per move later
            self.unexpanded_moves = list(legal_moves)

        return self.unexpanded_moves


    def add_child(self, move, prior_p=1.0):
        """
        Create and return a new child node for a given move.
        """
        child = MCTSNode(parent=self, prior_p=prior_p)
        self.children[move] = child
        return child


    def is_leaf(self):
        """
        A leaf node has no children yet.
        """
        return len(self.children) == 0


    def backprop(self, value):
        """
        Backpropagate a value up the tree.

        For a zero-sum game:
          If the current node gets +value,
          then parent gets -value (opponent perspective flips).

        value: result or NN value in [-1, +1]
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

        # Flip perspective for parent
        if self.parent is not None:
            self.parent.backprop(-value)
