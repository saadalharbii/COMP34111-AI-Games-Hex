import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.AgentBase import AgentBase
from src.Move import Move
from src.Colour import Colour

# We import utilities directly from the pretrained agent’s file
from agents.GroupX.TrainedAgent import OriginalModel, RotatedModel, create_border


class ExternalTrainedAgent(AgentBase):

    def __init__(self, colour):
        super().__init__(colour)
        self.colour = colour
        self.board_size = 11

        # ------------------------------
        # Load pretrained model
        # ------------------------------
        model_file_name = 'agents/GroupX/hex_ultra.pt'
        model_info = torch.load(model_file_name, map_location=torch.device('cpu'))

        model = OriginalModel(
            board_size=model_info['config'].getint('board_size'),
            layers=model_info['config'].getint('layers'),
            intermediate_channels=model_info['config'].getint('intermediate_channels'),
            reach=model_info['config'].getint('reach')
        )

        if model_info['config'].getboolean('rotation_model'):
            model = RotatedModel(model)

        model.load_state_dict(model_info['model_state_dict'])
        model.eval()

        self.model = model
        self.device = torch.device("cpu")


    # ---------------------------------------------------------
    # Convert engine board → tensor [2, 11, 11]
    # ---------------------------------------------------------
    def board_to_tensor(self, board):
        t = torch.zeros(2, self.board_size, self.board_size)

        for x in range(self.board_size):
            for y in range(self.board_size):
                c = board.tiles[x][y].colour
                if c == Colour.RED:
                    t[0, x, y] = 1
                elif c == Colour.BLUE:
                    t[1, x, y] = 1

        return t


    # ---------------------------------------------------------
    # Choose move using pretrained model + legal masking
    # ---------------------------------------------------------
    def choose_move(self, board):
        logical = self.board_to_tensor(board)

        # Add border as expected by model
        bordered = create_border(logical)
        inp = bordered.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(inp).squeeze(0)  # shape [121]

        # -----------------------------------------------------
        # FIX: Compute legal moves using engine board
        # -----------------------------------------------------
        legal_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board.tiles[x][y].colour is None:
                    legal_moves.append((x, y))

        # Mask all illegal moves
        masked_logits = torch.full((121,), float('-inf'))

        for (x, y) in legal_moves:
            idx = x * self.board_size + y
            masked_logits[idx] = logits[idx]

        # Pick legal move with highest score
        best_flat = torch.argmax(masked_logits).item()
        x = best_flat // self.board_size
        y = best_flat % self.board_size

        return x, y


    # ---------------------------------------------------------
    # Main engine API
    # ---------------------------------------------------------
    def make_move(self, turn, board, opp_move):
        x, y = self.choose_move(board)
        return Move(x, y)
