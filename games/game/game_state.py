import numpy as np
import random


class GameState:
    Move = object
    Payoff = np.ndarray

    def __init__(self):
        self.player = None
        self.player_count = None

    def moves(self) -> [Move]:
        raise NotImplementedError

    def apply_move(self, move: Move) -> None:
        raise NotImplementedError

    def undo_move(self, move: Move) -> None:
        raise NotImplementedError

    def is_final(self) -> bool:
        raise NotImplementedError

    def payoff(self) -> Payoff:
        raise NotImplementedError

    def as_matrix(self) -> np.array:
        raise NotImplementedError

    def random_playout_payoff(self) -> Payoff:
        stack = []

        while not self.is_final():
            move = random.choice(self.moves())
            self.apply_move(move)
            stack.append(move)

        payoff = self.payoff()

        while stack:
            move = stack.pop()
            self.undo_move(move)

        return payoff
