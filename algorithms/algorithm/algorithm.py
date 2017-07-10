from games.game.game_state import GameState
from games.game.move import Move

import math
import numpy as np


class Algorithm(object):
    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def move_rates(self) -> [(Move, np.array)]:
        raise NotImplementedError

    def best_move(self) -> Move:
        assert not self.game_state.is_final()

        best_move = None
        best_rate = -math.inf
        for move, rates in self.move_rates():
            if best_rate < rates[self.game_state.player]:
                best_move = move
                best_rate = rates[self.game_state.player]

        return best_move
