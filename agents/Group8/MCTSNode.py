# agents/Group8/MCTSNode.py

class MCTSNode:
    def __init__(self, parent=None, prior_p: float = 1.0):
        self.parent = parent

        # MCTS statistics
        self.P = prior_p   # prior probability (from NN)
        self.N = 0         # visit count
        self.W = 0.0       # total value
        self.Q = 0.0       # mean value

        # Children: dict[(x, y)] -> MCTSNode
        self.children = {}

        # For expansion: priors for each legal move
        self.priors = None  # dict[(x, y)] -> float or None

    # --------------------------------------
    def is_expanded(self) -> bool:
        """
        A node is 'expanded' once we've called expand() on it,
        i.e. once we computed its priors.
        """
        return self.priors is not None

    # --------------------------------------
    def expand(self, legal_moves, priors):
        """
        legal_moves: list[(x, y)]
        priors: dict[(x, y)] -> float
        Create child nodes for each legal move using provided priors.
        """
        self.priors = {}
        for move in legal_moves:
            p = priors.get(move, 1e-6)  # small fallback
            self.priors[move] = p
            if move not in self.children:
                self.children[move] = MCTSNode(parent=self, prior_p=p)

    # --------------------------------------
    def get_child(self, move):
        """
        Return existing child for move, or create it if it doesn't exist yet.
        """
        if move not in self.children:
            prior_p = 1e-6
            if self.priors is not None and move in self.priors:
                prior_p = self.priors[move]
            self.children[move] = MCTSNode(parent=self, prior_p=prior_p)
        return self.children[move]

    # --------------------------------------
    def backprop(self, value: float):
        """
        Backpropagate value up the tree.
        'value' is from the perspective of the player to move at THIS node.
        At each parent, the perspective flips (zero-sum).
        """
        node = self
        v = value

        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N

            # Flip perspective for parent
            v = -v
            node = node.parent
