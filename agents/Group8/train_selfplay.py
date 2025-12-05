# agents/Group8/train_selfplay.py

import math
import random
import os
from collections import deque
import time
start_time = time.time()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from src.Colour import Colour
from agents.Group8.FastBoard import FastBoard
from agents.Group8.Encoding import encode_board
from agents.Group8.PolicyValueNet import PolicyValueNet
from agents.Group8.MCTSNode import MCTSNode


# ============================================================
# CONFIG
# ============================================================

BOARD_SIZE = 11

NUM_SELFPLAY_GAMES = 400      # increase later (e.g. 1000+)
MCTS_SIMULATIONS = 256        # per move (tune up/down)
C_PUCT = 2.0

REPLAY_BUFFER_SIZE = 50000      
BATCH_SIZE = 256                

EPOCHS_PER_UPDATE = 4         # how many epochs per training round
LR = 1e-3
VALUE_LOSS_WEIGHT = 1.0
L2_WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "agents/Group8/model.pt"


# ============================================================
# MCTS for training (Fresh tree per move)
# ============================================================

def puct_score(parent, child, c_puct=C_PUCT):
    return child.Q + c_puct * child.P * math.sqrt(parent.N + 1) / (1 + child.N)


def nn_evaluate(model, board: FastBoard, current_colour: Colour):
    """
    Evaluate board with NN from current_colour's perspective.
    Return:
      priors: dict[(x,y)] -> prob (sum=1 over legal)
      value: float in [-1,1] from current_colour POV
    """
    winner = board.winner()
    if winner is not None:
        return {}, 1.0 if winner == current_colour else -1.0

    legal_moves = board.legal_moves()
    if not legal_moves:
        return {}, 0.0

    x = encode_board(board, current_colour, is_fastboard=True).to(DEVICE)
    with torch.no_grad():
        policy_logits, value = model(x)
    value = float(value.item())
    logits = policy_logits[0]  # (121,)

    # mask legal moves & renormalise
    indices = [mx * BOARD_SIZE + my for (mx, my) in legal_moves]
    legal_logits = logits[indices]
    probs = torch.softmax(legal_logits, dim=0).cpu().numpy()

    priors = {}
    for i, move in enumerate(legal_moves):
        priors[move] = float(probs[i])

    return priors, value


def mcts_search(model, root_board: FastBoard, player_colour: Colour, simulations=MCTS_SIMULATIONS):
    """
    Run MCTS from root_board for 'simulations' iterations.
    Returns:
      root: MCTSNode after search
    """
    root = MCTSNode(parent=None, prior_p=1.0)

    for _ in range(simulations):
        node = root
        board = root_board.copy()
        current_colour = player_colour

        # Selection
        while node.is_expanded() and node.children:
            # pick child by PUCT
            best_move, best_child = None, None
            best_score = -float("inf")
            for move, child in node.children.items():
                score = puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_child = child

            # play that move
            x, y = best_move
            board.play(current_colour, x, y)
            current_colour = Colour.RED if current_colour == Colour.BLUE else Colour.BLUE
            node = best_child

        # Expansion + evaluation (leaf)
        winner = board.winner()
        if winner is not None:
            v = 1.0 if winner == current_colour else -1.0
            node.backprop(v)
            continue

        legal_moves = board.legal_moves()
        if not legal_moves:
            node.backprop(0.0)
            continue

        priors, value = nn_evaluate(model, board, current_colour)

        # If priors ended up empty, fallback to uniform
        if not priors:
            p = 1.0 / len(legal_moves)
            priors = {m: p for m in legal_moves}

        node.expand(legal_moves, priors)
        node.backprop(value)

    return root


def root_policy_from_tree(root: MCTSNode):
    """
    Convert root children's visit counts into a π distribution over 121 moves.
    """
    # Build policy over all board positions
    pi = torch.zeros(BOARD_SIZE * BOARD_SIZE, dtype=torch.float32)

    total_N = sum(child.N for child in root.children.values())
    if total_N == 0:
        # no visits? uniform
        return pi + 1.0 / (BOARD_SIZE * BOARD_SIZE)

    for (x, y), child in root.children.items():
        idx = x * BOARD_SIZE + y
        pi[idx] = child.N / total_N

    return pi


# ============================================================
# Self-play game generation
# ============================================================

def play_selfplay_game(model):
    """
    Play one self-play game using MCTS+NN.
    Returns:
      states: list[Tensor  (2,11,11)]
      policies: list[Tensor (121,)]
      rewards: list[float]  (+1 / -1 for each state, from that state's player POV)
    """
    board = FastBoard(BOARD_SIZE)
    current_colour = Colour.RED

    states = []
    policies = []
    players = []   # which colour acted at each state

    while True:
        # encode current state
        x = encode_board(board, current_colour, is_fastboard=True)[0]  # (2,11,11)
        states.append(x)
        players.append(current_colour)

        # run MCTS from this state
        root = mcts_search(model, board, current_colour, simulations=MCTS_SIMULATIONS)

        # get π from visit counts
        pi = root_policy_from_tree(root)
        policies.append(pi)

        # sample move from π (for exploration) or argmax
        if random.random() < 0.25:
            # exploration: sample by π
            flat_idx = torch.multinomial(pi, num_samples=1).item()
        else:
            # exploitation: argmax
            flat_idx = torch.argmax(pi).item()

        x_move = flat_idx // BOARD_SIZE
        y_move = flat_idx % BOARD_SIZE

        board.play(current_colour, x_move, y_move)

        # check for end
        winner = board.winner()
        if winner is not None:
            # assign rewards z_i to each state
            rewards = []
            for p in players:
                rewards.append(1.0 if p == winner else -1.0)
            return states, policies, rewards

        # switch player
        current_colour = Colour.RED if current_colour == Colour.BLUE else Colour.BLUE


# ============================================================
# Training loop
# ============================================================

def train():
    print("Using device:", DEVICE)

    model = PolicyValueNet(board_size=BOARD_SIZE).to(DEVICE)

    # Load existing model if present
    if os.path.exists(MODEL_PATH):
        print("Loading existing model from", MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print("No existing model found, training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_WEIGHT_DECAY)
    scaler = GradScaler()

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    model.train()

    game_count = 0

    for game_idx in range(NUM_SELFPLAY_GAMES):
        print(f"\n=== Self-play game {game_idx + 1}/{NUM_SELFPLAY_GAMES} ===")
        elapsed = time.time() - start_time
        print(f"Elapsed training time: {elapsed/60:.2f} minutes")

        states, policies, rewards = play_selfplay_game(model)

        # push into replay buffer
        for s, pi, z in zip(states, policies, rewards):
            replay_buffer.append((s, pi, z))

        game_count += 1
        print(f"Replay buffer size: {len(replay_buffer)}")

        # train every few games once buffer is big enough
        if len(replay_buffer) >= BATCH_SIZE:
            for epoch in range(EPOCHS_PER_UPDATE):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                s_batch, pi_batch, z_batch = zip(*batch)

                s_batch = torch.stack(s_batch).to(DEVICE)            # (B, 2,11,11)
                pi_batch = torch.stack(pi_batch).to(DEVICE)          # (B, 121)
                z_batch = torch.tensor(z_batch, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # (B,1)

                optimizer.zero_grad()

                with autocast(enabled=DEVICE.type == "cuda"):
                    policy_logits, values = model(s_batch)           # logits:(B,121), values:(B,1)

                    # Policy loss: cross-entropy between target π and logits
                    log_probs = torch.log_softmax(policy_logits, dim=1)
                    policy_loss = -torch.mean(torch.sum(pi_batch * log_probs, dim=1))

                    # Value loss: MSE
                    value_loss = torch.mean((values - z_batch) ** 2)

                    loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                print(f"Game {game_idx+1}, epoch {epoch+1}: "
                      f"loss={loss.item():.4f}, "
                      f"policy_loss={policy_loss.item():.4f}, "
                      f"value_loss={value_loss.item():.4f}")

            # Save checkpoint
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save({"model": model.state_dict()}, MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
