import numpy as np


class GameState:
    Move = int
    Payoff = np.ndarray

    def __init__(self):
        self.player = None
        self.player_count = None

    def move_count(self) -> int:
        raise NotImplementedError

    def get_move(self, index: int):
        raise NotImplementedError

    def apply_move(self, index: int) -> int:
        raise NotImplementedError

    def undo_move(self) -> int:
        raise NotImplementedError

    def is_final(self) -> bool:
        raise NotImplementedError

    def payoff(self) -> np.ndarray:
        raise NotImplementedError

    def as_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def random_playout_payoff(self) -> np.ndarray:
        raise NotImplementedError
