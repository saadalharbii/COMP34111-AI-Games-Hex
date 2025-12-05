# agents/Group8/Encoding.py

import torch
from src.Colour import Colour


def encode_board(board, our_colour, is_fastboard=True):
    """
    Encode either:
      - FastBoard (bitboards)
      - Engine Board (tiles[x][y])

    Output shape: (1, 2, size, size)

    Channel 0 = our stones
    Channel 1 = opponent stones
    """
    s = board.size
    x = torch.zeros((1, 2, s, s), dtype=torch.float32)

    opp_colour = Colour.RED if our_colour == Colour.BLUE else Colour.BLUE

    # ----------------------------------------------
    # Case 1: FastBoard (bitboards)
    # ----------------------------------------------
    if is_fastboard:
        red_bits = board.red_bits
        blue_bits = board.blue_bits

        for idx in range(s * s):
            xi = idx // s
            yi = idx % s
            mask = 1 << idx

            # Our stones
            if our_colour == Colour.RED:
                if red_bits & mask:
                    x[0, 0, xi, yi] = 1.0
            elif our_colour == Colour.BLUE:
                if blue_bits & mask:
                    x[0, 0, xi, yi] = 1.0

            # Opponent stones
            if opp_colour == Colour.RED:
                if red_bits & mask:
                    x[0, 1, xi, yi] = 1.0
            elif opp_colour == Colour.BLUE:
                if blue_bits & mask:
                    x[0, 1, xi, yi] = 1.0

        return x

    # ----------------------------------------------
    # Case 2: engine Board (tiles[x][y])
    # ----------------------------------------------
    else:
        for xi in range(s):
            for yi in range(s):
                tile = board.tiles[xi][yi]

                if tile.colour == our_colour:
                    x[0, 0, xi, yi] = 1.0
                elif tile.colour == opp_colour:
                    x[0, 1, xi, yi] = 1.0

        return x
