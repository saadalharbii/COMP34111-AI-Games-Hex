# agents/Group8/MyAgent.py

import math
import time
import random

from agents.Group8.FastBoard import FastBoard
from agents.Group8.MCTSNode import MCTSNode
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase


# ============================
#    PUCT SCORE FUNCTION
# ============================
def puct_score(parent, child, c_puct=1.5):
    """
    PUCT formula used by AlphaZero-style MCTS.
    Works even if P = 1 (uniform priors before NN).
    """
    prior = child.P

    if child.N == 0:
        u = c_puct * prior * math.sqrt(parent.N + 1)
    else:
        u = c_puct * prior * math.sqrt(parent.N) / (1 + child.N)

    return child.Q + u


# ============================
#        MyAgent CLASS
# ============================
class MyAgent(AgentBase):

    def __init__(self, colour):
        super().__init__(colour)
        self.colour = colour
        self.board_size = 11

        # Opponent colour tracking
        self.opponent_colour = Colour.RED if colour == Colour.BLUE else Colour.BLUE

        # Fast internal board
        self.fast_board = FastBoard(self.board_size)

        # MCTS root node
        self.root = MCTSNode(parent=None)

        self.swap_used = False
        self.total_time_used = 0

        # Time control
        self.time_limit = 0.35  # seconds per move
        self.max_total_time = 280  # total seconds (remaining 20s safety)

    # ==========================================================
    #               Board Sync Functions
    # ==========================================================
    def update_fast_board(self, board):
        """Rebuild fastboard from the engine board."""
        fb = FastBoard(board.size)

        for x in range(board.size):
            for y in range(board.size):
                tile = board.tiles[x][y]
                if tile.colour == Colour.RED:
                    fb.play(Colour.RED, x, y)
                elif tile.colour == Colour.BLUE:
                    fb.play(Colour.BLUE, x, y)

        self.fast_board = fb

    def apply_move_to_fast(self, x, y, colour):
        """Apply move internally."""
        ok = self.fast_board.play(colour, x, y)
        assert ok, f"Illegal move applied to fast board: {(x, y)}"

    def get_legal_moves(self):
        return self.fast_board.legal_moves()

    # ==========================================================
    #               Swap Decision (Pie Rule)
    # ==========================================================
    def decide_swap(self, board, opp_move):
        """
        Decide whether to use the swap (pie rule) as BLUE on turn 2.

        Heuristic:
        - If opponent's first move is near the centre, we swap.
        - If it's far from centre (weak opening), we keep BLUE.
        """

        if opp_move is None or opp_move.x < 0 or opp_move.y < 0:
            return False

        size = board.size
        cx = (size - 1) / 2.0
        cy = (size - 1) / 2.0

        dx = abs(opp_move.x - cx)
        dy = abs(opp_move.y - cy)

        # Central 5x5 zone (max Chebyshev distance <= 2)
        if max(dx, dy) <= 2:
            return True

        return False

    # ==========================================================
    #               MCTS: SELECTION (PUCT)
    # ==========================================================
    def select_child(self, node):
        best_move = None
        best_child = None
        best_score = -float('inf')

        for move, child in node.children.items():
            score = puct_score(node, child)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    # ==========================================================
    #             MCTS: TEMPORARY VALUE FUNCTION
    # ==========================================================
    def evaluate_state(self, board):
        winner = board.winner()
        if winner is None:
            return 0
        return 1 if winner == self.colour else -1

    # ==========================================================
    #               MCTS: ONE SIMULATION
    # ==========================================================
    def run_mcts_simulation(self):
        node = self.root
        board = self.fast_board.copy()
        current_colour = self.colour

        # 1. SELECTION
        while node.is_expanded() and node.unexpanded_moves == []:
            move, node = self.select_child(node)
            x, y = move
            board.play(current_colour, x, y)
            current_colour = Colour.RED if current_colour == Colour.BLUE else Colour.BLUE

        # 2. EXPANSION
        legal_moves = board.legal_moves()

        if not node.is_expanded():
            node.expand(legal_moves)

        if node.unexpanded_moves:
            move = node.unexpanded_moves.pop()
            child = node.add_child(move)
            x, y = move
            board.play(current_colour, x, y)
            node = child
            current_colour = Colour.RED if current_colour == Colour.BLUE else Colour.BLUE

        # 3. EVALUATION
        value = self.evaluate_state(board)

        # 4. BACKPROP
        node.backprop(value)

    # ==========================================================
    #               MCTS SEARCH ENTRY POINT
    # ==========================================================
    def mcts_search(self, turn):
        start_time = time.time()
        time_limit = self.time_limit

        # Panic mode if running out of total time
        if self.total_time_used / 1e9 > self.max_total_time:
            time_limit = 0.05

        while (time.time() - start_time) < time_limit:
            self.run_mcts_simulation()

        # If no children (extremely rare), pick random move
        if len(self.root.children) == 0:
            return random.choice(self.fast_board.legal_moves())

        # Choose child with most visits
        best_move = max(
            self.root.children.items(),
            key=lambda item: item[1].N
        )[0]

        return best_move

    # ==========================================================
    #                     MAIN ENTRY
    # ==========================================================
    def make_move(self, turn, board, opp_move):
        start = time.time_ns()

        # 1. Sync fastboard
        self.update_fast_board(board)

        # 1a. Panic check AFTER sync
        if (self.total_time_used / 1e9) > self.max_total_time:
            x, y = random.choice(self.fast_board.legal_moves())
            self.apply_move_to_fast(x, y, self.colour)
            self.advance_tree_after_our_move((x, y))
            return Move(x, y)

        # 1b. Swap rule: only BLUE on turn 2 can use pie rule
        if turn == 2 and self.colour == Colour.BLUE and not self.swap_used:
            if self.decide_swap(board, opp_move):
                self.swap_used = True
                end = time.time_ns()
                self.total_time_used += (end - start)
                # Swap move is represented by (-1, -1)
                return Move(-1, -1)

        # 1c. Opponent move → reuse tree (board already updated by engine)
        if opp_move is not None and opp_move.x != -1:
            self.advance_tree_after_opponent_move((opp_move.x, opp_move.y))

        # 2. Choose move via MCTS
        x, y = self.mcts_search(turn)

        # 3. Apply move internally
        self.apply_move_to_fast(x, y, self.colour)

        # 4. Tree reuse
        self.advance_tree_after_our_move((x, y))

        end = time.time_ns()
        self.total_time_used += (end - start)

        return Move(x, y)

    # ==========================================================
    #                 TREE REUSE FUNCTIONS
    # ==========================================================
    def advance_tree_after_our_move(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = MCTSNode(parent=None)

    def advance_tree_after_opponent_move(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = MCTSNode(parent=None)
