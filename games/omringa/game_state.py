from enum import Enum
from games.game import game_state
from recordclass import recordclass

import numpy as np


class GameState(game_state.GameState):
    class State(Enum):
        bet = 0
        nature = 1
        place = 2

    Move = recordclass('Move', 'state player value idx')

    def __init__(self, board_size, group_penalty=-5, min_bet=0, max_bet=9):
        self.state = GameState.State.bet
        self.bets = [None, None]
        self.chosen_player = None
        self.board = np.zeros((board_size, board_size), dtype=np.int)
        self.empty_positions = [(y, x) for y in range(board_size) for x in range(board_size)]

        self.board_size = board_size
        self.group_penalty = group_penalty
        self.min_bet = min_bet
        self.max_bet = max_bet

    def moves(self) -> [Move]:
        if self.state == GameState.State.bet:
            return [GameState.Move(self.state, self.player, value, None)
                    for value in range(self.min_bet, self.max_bet)]
        elif self.state == GameState.State.nature:
            return [GameState.Move(self.state, self.player, player, None)
                    for player in range(2)]
        else:
            return [GameState.Move(self.state, self.player, value, idx)
                    for idx, value in enumerate(self.empty_positions)]

    def apply_move(self, move: Move) -> None:
        assert move.state == self.state
        assert move.player == self.player

        if self.state == GameState.State.bet:
            self.bets[move.player] = move.value
            if self.bets[move.player ^ 1] is None:
                self.state = GameState.State.bet
                self.player = self.player ^ 1
            elif self.bets[0] == self.bets[1]:
                self.player = GameState.State.nature
                self.player = -1
            else:
                self.state = GameState.State.place
                self.player = np.argmax(self.bets)
        elif self.state == GameState.State.nature:
            self.state = GameState.State.place
            self.player = move.value
            self.chosen_player = move.value
        else:
            self.empty_positions[move.idx], self.empty_positions[-1] = \
                self.empty_positions[-1], self.empty_positions[move.idx]
            self.empty_positions = self.empty_positions[:-1]
            self.board[move.value[0]][move.value[1]] = move.player + 1
            self.player = move.player ^ 1

    def undo_move(self, move: Move) -> None:
        self.state = move.state
        self.player = move.player
        if move.state == GameState.State.bet:
            self.bets[move.player] = None
        elif move.state == GameState.State.nature:
            self.chosen_player = None
        else:
            self.empty_positions += [move.value]
            self.empty_positions[move.idx], self.empty_positions[-1] = \
                self.empty_positions[-1], self.empty_positions[move.idx]
            self.board[move.value[0]][move.value[1]] = 0

    def is_final(self) -> bool:
        return self.empty_positions == []

    def count_groups(self, player):
        player = player + 1
        result = 0

        visited = np.zeros((self.board_size, self.board_size), dtype=np.int)
        for p in self.empty_positions:
            visited[p[0]][p[1]] = 1

        for y in range(self.board_size):
            for x in range(self.board_size):
                if visited[y][x] == 0 and self.board[y][x] == player:
                    visited[y][x] = 1
                    result += 1

                    stack = [(y, x)]
                    while stack is not []:
                        p, stack = stack[-1], stack[:-1]
                        for dy in [-1, 0, 0, 1]:
                            for dx in [0, -1, 1, 0]:
                                if visited[y + dy][x + dx] == 0 and self.board[y + dy][x + dx] == player:
                                    visited[y + dy][x + dx] = 1
                                    stack += [(y + dy, x + dx)]

        return result

    def payoff(self) -> np.array:
        result = np.array(
            [[self.count_groups(player) * self.group_penalty]
             for player in range(2)],
            dtype=np.float32)

        if self.bets[0] < self.bets[1]:
            order = [1, 0]
        elif self.bets[0] > self.bets[1]:
            order = [0, 1]
        else:
            order = [self.chosen_player, self.chosen_player ^ 1]

        result[order[0]][0] += np.ceil(self.board_size * self.board_size / 2)
        result[order[1]][0] += np.floor(self.board_size * self.board_size / 2)
        result[order[1]][0] += self.bets[order[1]] + 0.5

        return result

    def __deepcopy__(self, memodict={}):
        game_state = GameState(self.board_size, self.group_penalty, self.min_bet, self.max_bet)
        game_state.state = self.state
        game_state.bets = list(self.bets)
        game_state.chosen_player = self.chosen_player
        game_state.board = np.array(self.board, copy=True)
        game_state.empty_positions = list(self.empty_positions)
