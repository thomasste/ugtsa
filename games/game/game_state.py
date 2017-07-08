from games.game.move import Move
from numpy import array


class GameState:
    def __init__(self):
        self.player = None

    def moves(self) -> [Move]:
        raise NotImplementedError

    def apply_move(self, move: Move) -> None:
        raise NotImplementedError

    def undo_move(self, move: Move) -> None:
        raise NotImplementedError

    def is_final(self) -> bool:
        raise NotImplementedError

    def payoff(self) -> array:
        raise NotImplementedError
