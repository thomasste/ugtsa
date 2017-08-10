# distutils: language = c++

from games.omringa cimport game_state
from libc.stdlib cimport rand
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np


cdef class GameState:
    def __init__(self, int board_size=7, int group_penalty=-5, int min_bet=0, int max_bet=9):
        cdef Position position

        self.player = 0
        self.player_count = 2
        self.board_size = board_size
        self.group_penalty = group_penalty
        self.min_bet = min_bet
        self.max_bet = max_bet

        self.state = State.bet
        self.bets[0] = -1
        self.bets[1] = -1
        self.chosen_player = -1
        self.board = np.zeros((board_size, board_size), dtype=np.int)

        for y in range(board_size):
            for x in range(board_size):
                position.y = y
                position.x = x
                self.empty_positions.push_back(position)

    cpdef int move_count(self) except *:
        if self.state == State.bet:
            return self.max_bet - self.min_bet
        elif self.state == State.nature:
            return 2
        else:
            return self.empty_positions.size()

    cpdef Move get_move(self, int index) except *:
        cdef Move move
        move.state = self.state
        move.player = self.player

        if self.state == State.bet:
            move.value[0] = self.min_bet + index
            move.index = -1
        elif self.state == State.nature:
            move.value[0] = index
            move.index = -1
        else:
            move.value[0] = self.empty_positions[index].y
            move.value[1] = self.empty_positions[index].x
            move.index = index
        return move

    cpdef void apply_move(self, int index) except *:
        cdef Move move = self.get_move(index)
        self.move_history.push_back(move)

        if self.state == State.bet:
            self.bets[move.player] = move.value[0]
            if self.bets[move.player ^ 1] == -1:
                self.state = State.bet
                self.player = self.player ^ 1
            elif self.bets[0] == self.bets[1]:
                self.state = State.nature
                self.player = -1
            else:
                self.state = State.place
                self.player = 0 if self.bets[0] > self.bets[1] else 1
        elif self.state == State.nature:
            self.state = State.place
            self.player = move.value[0]
            self.chosen_player = move.value[0]
        else:
            self.empty_positions[move.index] = self.empty_positions.back()
            self.empty_positions.pop_back()
            self.board[move.value[0]][move.value[1]] = move.player + 1
            self.player = move.player ^ 1

    cpdef void undo_move(self) except *:
        cdef Move move = self.move_history.back()
        self.move_history.pop_back()

        cdef Position empty_position
        empty_position.y = move.value[0]
        empty_position.x = move.value[1]

        self.state = move.state
        self.player = move.player
        if move.state == State.bet:
            self.bets[move.player] = -1
        elif move.state == State.nature:
            self.chosen_player = -1
        else:
            self.empty_positions.push_back(self.empty_positions[move.index])
            self.empty_positions[move.index] = empty_position
            self.board[move.value[0]][move.value[1]] = 0

    cpdef bool is_final(self) except *:
        return self.empty_positions.empty()

    cpdef int count_groups(self, int player) except *:
        player = player + 1
        cpdef int result = 0
        cpdef vector[Position] stack
        cpdef Position position1, position2

        cdef np.ndarray[np.int32_t, ndim=2] visited = np.zeros((self.board_size, self.board_size), dtype=np.int32)

        for y in range(self.board_size):
            for x in range(self.board_size):
                if visited[y][x] == 0 and self.board[y][x] == player:
                    visited[y][x] = 1
                    result += 1

                    position1.y = y
                    position1.x = x
                    stack.push_back(position1)
                    while not stack.empty():
                        position1 = stack.back()
                        stack.pop_back()
                        for dy, dx in zip([-1, 0, 0, 1], [0, -1, 1, 0]):
                            ny = position1.y + dy
                            nx = position1.x + dx
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and \
                                    visited[ny][nx] == 0 and self.board[ny][nx] == player:
                                visited[ny][nx] = 1
                                position2.y = ny
                                position2.x = nx
                                stack.push_back(position2)

        return result

    cpdef np.ndarray[np.float32_t, ndim=1] payoff(self):
        result = np.array([
            self.count_groups(0) * self.group_penalty,
            self.count_groups(1) * self.group_penalty
            ], dtype=np.float32)

        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == 1:
                    result[0] += 1
                elif self.board[y][x] == 2:
                    result[1] += 1

        if self.bets[0] < self.bets[1]:
            result[0] += self.bets[0] + 0.5
        elif self.bets[0] > self.bets[1]:
            result[1] += self.bets[1] + 0.5
        else:
            result[self.chosen_player ^ 1] += self.bets[self.chosen_player ^ 1] + 0.5

        return result

    cpdef np.ndarray[np.float32_t, ndim=2] as_matrix(self):
        return np.array(self.board, copy=True, dtype=np.float32)

    cpdef np.ndarray[np.float32_t, ndim=1] random_playout_payoff(self):
        cdef counter = 0

        while not self.is_final():
            counter += 1
            self.apply_move(rand() % self.move_count())

        cdef np.ndarray[np.float32_t, ndim=1] payoff = self.payoff()

        for i in range(counter):
            self.undo_move()

        return payoff

    def __str__(self):
        lines = []
        lines.append('player {}'.format(self.player))
        lines.append('chosen_player {}'.format(self.chosen_player))
        lines.append('state {}'.format(self.state))
        lines.append('bets {} {}'.format(self.bets[0], self.bets[1]))
        lines.append(str(self.board))
        return '\n'.join(lines)

    cpdef GameState __deepcopy__(self, memodict):
        cdef GameState game_state = GameState(
            self.board_size, self.group_penalty, self.min_bet, self.max_bet)

        game_state.player = self.player

        game_state.state = self.state
        game_state.bets[0] = self.bets[0]
        game_state.bets[1] = self.bets[1]
        game_state.chosen_player = self.chosen_player
        game_state.board = np.array(self.board, copy=True)
        game_state.empty_positions = vector[Position](self.empty_positions)
        game_state.move_history = vector[Move](self.move_history)
        return game_state