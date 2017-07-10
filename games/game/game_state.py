from numpy import array


class GameState:
    Move = object

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

    def payoff(self) -> array:
        raise NotImplementedError
