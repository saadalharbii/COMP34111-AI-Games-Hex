from src.Colour import Colour

############################################################
# DSU (Union-Find)
############################################################

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        # Path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


############################################################
# FastBoard for Hex
############################################################

class FastBoard:
    """
    Internal fast Hex board for MCTS/NN.

    Coordinate convention is aligned with engine:

        engine: board.tiles[x][y]
        here:   index = x * size + y

    So:
        x in [0, size)
        y in [0, size)

    Representation:
      - red_bits / blue_bits: bitboard occupancy
      - uf_red / uf_blue: DSU with 2 virtual nodes each
        * Red:  top ↔ bottom
        * Blue: left ↔ right
    """

    def __init__(self, size: int = 11):
        self.size = size
        self.N = size * size  # 121 for 11x11

        # Bitboards
        self.red_bits = 0
        self.blue_bits = 0

        # Virtual DSU nodes
        self.RED_TOP = self.N
        self.RED_BOTTOM = self.N + 1
        self.BLUE_LEFT = self.N + 2
        self.BLUE_RIGHT = self.N + 3

        # DSUs per color
        self.uf_red = DSU(self.N + 4)
        self.uf_blue = DSU(self.N + 4)

        # Precomputed neighbors by index (x-major)
        self.neighbors: list[list[int]] = [[] for _ in range(self.N)]
        self._compute_neighbors()

    ############################################################
    # Neighbors
    ############################################################

    def _compute_neighbors(self) -> None:
        """Precompute 6 hex neighbors for each index, using x-major layout."""
        s = self.size
        for x in range(s):
            for y in range(s):
                i = x * s + y
                for dx, dy in [(-1, 0), (1, 0),
                               (0, -1), (0, 1),
                               (-1, 1), (1, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < s and 0 <= ny < s:
                        j = nx * s + ny
                        self.neighbors[i].append(j)

    ############################################################
    # Legality
    ############################################################

    def is_legal(self, x: int, y: int) -> bool:
        """Square is empty? (aligned with engine Board tiles[x][y])."""
        idx = x * self.size + y
        mask = 1 << idx
        occ = self.red_bits | self.blue_bits
        return not (occ & mask)

    ############################################################
    # Play move
    ############################################################

    def play(self, colour_enum: Colour, x: int, y: int) -> bool:
        """
        Apply a move for RED or BLUE at (x, y).
        Returns False if the move is illegal (occupied).
        """

        if not self.is_legal(x, y):
            return False

        idx = x * self.size + y
        mask = 1 << idx

        if colour_enum == Colour.RED:
            self.red_bits |= mask
            self._union_red(idx, x, y)
        else:
            self.blue_bits |= mask
            self._union_blue(idx, x, y)

        return True

    ############################################################
    # DSU connections
    ############################################################

    def _union_red(self, i: int, x: int, y: int) -> None:
        # Connect to virtual edges: top (y=0) / bottom (y=size-1)
        if y == 0:
            self.uf_red.union(i, self.RED_TOP)
        if y == self.size - 1:
            self.uf_red.union(i, self.RED_BOTTOM)

        # Connect to same-color neighbors
        for j in self.neighbors[i]:
            if (self.red_bits >> j) & 1:
                self.uf_red.union(i, j)

    def _union_blue(self, i: int, x: int, y: int) -> None:
        # Connect to virtual edges: left (x=0) / right (x=size-1)
        if x == 0:
            self.uf_blue.union(i, self.BLUE_LEFT)
        if x == self.size - 1:
            self.uf_blue.union(i, self.BLUE_RIGHT)

        # Connect to same-color neighbors
        for j in self.neighbors[i]:
            if (self.blue_bits >> j) & 1:
                self.uf_blue.union(i, j)

    ############################################################
    # Win detection
    ############################################################

    def check_red_win(self) -> bool:
        return self.uf_red.find(self.RED_TOP) == self.uf_red.find(self.RED_BOTTOM)

    def check_blue_win(self) -> bool:
        return self.uf_blue.find(self.BLUE_LEFT) == self.uf_blue.find(self.BLUE_RIGHT)

    def winner(self):
        if self.check_red_win():
            return Colour.RED
        if self.check_blue_win():
            return Colour.BLUE
        return None

    ############################################################
    # Legal moves enumeration
    ############################################################

    def legal_moves(self) -> list[tuple[int, int]]:
        """Return list of all (x, y) that are empty."""
        occ = self.red_bits | self.blue_bits
        moves: list[tuple[int, int]] = []
        for idx in range(self.N):
            if not ((occ >> idx) & 1):
                x = idx // self.size
                y = idx % self.size
                moves.append((x, y))
        return moves

    ############################################################
    # Copy (for MCTS branching)
    ############################################################

    def copy(self) -> "FastBoard":
        fb = FastBoard(self.size)
        fb.red_bits = self.red_bits
        fb.blue_bits = self.blue_bits

        fb.uf_red.parent = self.uf_red.parent[:]
        fb.uf_red.rank   = self.uf_red.rank[:]
        fb.uf_blue.parent = self.uf_blue.parent[:]
        fb.uf_blue.rank   = self.uf_blue.rank[:]

        # neighbors, size, and virtual node indices are identical / shared
        return fb
