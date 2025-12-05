# agents/Group8/MyAgent.py

import math
import time
import random
import os
import torch

from agents.Group8.FastBoard import FastBoard
from agents.Group8.MCTSNode import MCTSNode
from agents.Group8.Encoding import encode_board
from agents.Group8.PolicyValueNet import PolicyValueNet

from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase


def puct_score(parent: MCTSNode, child: MCTSNode, c_puct: float = 1.5) -> float:
    """
    Standard PUCT formula:
        Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    """
    return child.Q + c_puct * child.P * math.sqrt(parent.N + 1) / (1 + child.N)


class MyAgent(AgentBase):

    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.colour = colour
        self.opponent_colour = Colour.RED if colour == Colour.BLUE else Colour.BLUE

        self.board_size = 11
        self.fast_board = FastBoard(self.board_size)

        # Root of the MCTS tree
        self.root = MCTSNode(parent=None, prior_p=1.0)

        # Time management
        self.total_time_used = 0
        self.time_limit = 0.35       # per move (seconds)
        self.max_total_time = 280.0  # total per game (seconds), with safety margin

        # Device & NN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.nn_enabled = False

        model_path = "agents/Group8/model.pt"
        if os.path.exists(model_path):
            try:
                print("[MyAgent] Loading model from", model_path)
                self.model = PolicyValueNet(board_size=self.board_size).to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                # Accept either full checkpoint dict or raw state_dict
                state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.nn_enabled = True
                print("[MyAgent] Model loaded. NN-guided MCTS enabled.")
            except Exception as e:
                print("[MyAgent] Failed to load model, running pure MCTS:", e)
                self.model = None
                self.nn_enabled = False
        else:
            print("[MyAgent] No model found - running pure MCTS mode.")
            self.model = None
            self.nn_enabled = False

    # ==========================================================
    # Board sync
    # ==========================================================
    def update_fast_board(self, board):
        """
        Rebuild FastBoard from engine Board.
        """
        fb = FastBoard(board.size)
        for x in range(board.size):
            for y in range(board.size):
                tile = board.tiles[x][y]
                if tile.colour == Colour.RED:
                    fb.play(Colour.RED, x, y)
                elif tile.colour == Colour.BLUE:
                    fb.play(Colour.BLUE, x, y)
        self.fast_board = fb

    # ==========================================================
    # NN evaluation (policy + value)
    # ==========================================================
    def nn_evaluate(self, board: FastBoard, current_colour: Colour):
        """
        Evaluate a FastBoard position from the perspective of 'current_colour'.
        Returns:
            priors: dict[(x, y)] -> float  (sum to 1 over legal moves)
            value: float in [-1, 1] (current player's perspective)
        """
        # Terminal check first (important)
        winner = board.winner()
        if winner is not None:
            if winner == current_colour:
                return {}, 1.0
            else:
                return {}, -1.0

        legal_moves = board.legal_moves()
        if not legal_moves:
            # No moves, treat as neutral
            return {}, 0.0

        # Encode board from current player's perspective
        x = encode_board(board, current_colour, is_fastboard=True).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(x)  # value: shape (1, 1)
        value = float(value.item())

        logits = policy_logits[0]  # shape (121,)

        # Mask illegal moves and renormalize over LEGAL only
        indices = []
        for (mx, my) in legal_moves:
            idx = mx * self.board_size + my
            indices.append(idx)

        legal_logits = logits[indices]
        probs = torch.softmax(legal_logits, dim=0).cpu().numpy()

        priors = {}
        for i, move in enumerate(legal_moves):
            priors[move] = float(probs[i])

        return priors, value

    # ==========================================================
    # MCTS Selection
    # ==========================================================
    def select_child(self, node: MCTSNode):
        best_move = None
        best_child = None
        best_score = -float("inf")

        for move, child in node.children.items():
            score = puct_score(node, child)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    # ==========================================================
    # One MCTS simulation
    # ==========================================================
    def run_mcts_simulation(self):
        node = self.root
        board = self.fast_board.copy()
        current_colour = self.colour

        # 1. Selection: descend while node is expanded and has children
        while node.is_expanded() and node.children:
            move, child = self.select_child(node)
            x, y = move
            board.play(current_colour, x, y)
            current_colour = Colour.RED if current_colour == Colour.BLUE else Colour.BLUE
            node = child

        # 2. Expansion + Evaluation / Terminal handling
        winner = board.winner()
        if winner is not None:
            # Value from perspective of the player to move (current_colour)
            value = 1.0 if winner == current_colour else -1.0
            node.backprop(value)
            return

        legal_moves = board.legal_moves()
        if not legal_moves:
            node.backprop(0.0)
            return

        # Evaluate with NN (if enabled), otherwise use uniform priors and 0 value
        if self.nn_enabled and self.model is not None:
            priors, value = self.nn_evaluate(board, current_colour)
            if not priors:
                p = 1.0 / len(legal_moves)
                priors = {m: p for m in legal_moves}
        else:
            p = 1.0 / len(legal_moves)
            priors = {m: p for m in legal_moves}
            value = 0.0

        # Expand node using priors (children created here)
        node.expand(legal_moves, priors)

        # Backprop value from this leaf
        node.backprop(value)

    # ==========================================================
    # MCTS search wrapper
    # ==========================================================
    def mcts_search(self, turn: int):
        start = time.time()
        time_limit = self.time_limit

        # Panic mode if we're close to total time limit
        if (self.total_time_used / 1e9) > self.max_total_time:
            time_limit = 0.05

        while (time.time() - start) < time_limit:
            self.run_mcts_simulation()

        if not self.root.children:
            # No search info? just pick a random legal move
            legal = self.fast_board.legal_moves()
            return random.choice(legal)

        # Choose move with highest visit count
        best_move = max(self.root.children.items(), key=lambda item: item[1].N)[0]
        return best_move

    # ==========================================================
    # Tree reuse after moves
    # ==========================================================
    def advance_tree_after_move(self, move):
        """
        Re-root the tree at the given move, if it exists.
        Otherwise, reset tree.
        """
        if move in self.root.children:
            new_root = self.root.children[move]
            new_root.parent = None
            self.root = new_root
        else:
            self.root = MCTSNode(parent=None, prior_p=1.0)

    # ==========================================================
    # Swap (pie rule) heuristic
    # ==========================================================
    def should_swap(self, turn, opp_move):
        """
        Simple heuristic swap rule:
        As Blue on turn 2, if Red played near the center, we swap.
        """
        if turn != 2:
            return False
        if self.colour != Colour.BLUE:
            return False
        if opp_move is None or opp_move.x == -1:
            return False

        x, y = opp_move.x, opp_move.y

        # Central region heuristic
        if 3 <= x <= 7 and 3 <= y <= 7:
            return True

        return False

    # ==========================================================
    # Main entry point
    # ==========================================================
    def make_move(self, turn, board, opp_move):
        start = time.time_ns()

        # 1. Swap rule (Blue, turn 2) - based only on opp move
        if self.should_swap(turn, opp_move):
            end = time.time_ns()
            self.total_time_used += (end - start)
            return Move(-1, -1)  # swap move

        # 2. Sync internal fast board from engine
        self.update_fast_board(board)

        # 3. Tree reuse after opponent move (DO NOT re-apply move to fast_board)
        if opp_move is not None and opp_move.x != -1:
            self.advance_tree_after_move((opp_move.x, opp_move.y))

        # 4. Run MCTS (NN-guided if model is loaded)
        x, y = self.mcts_search(turn)

        # 5. Re-root tree at our chosen move
        self.advance_tree_after_move((x, y))

        end = time.time_ns()
        self.total_time_used += (end - start)

        return Move(x, y)
