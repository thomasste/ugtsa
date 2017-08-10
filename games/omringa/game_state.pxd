# distutils: language = c++

from libc.stdlib cimport rand
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np

cdef struct Position:
    int x
    int y

cdef enum State:
    bet = 0
    nature = 1
    place = 2

cdef struct Move:
    State state
    int player
    int value[2]
    int index

cdef class GameState:
    cdef readonly int player
    cdef readonly int player_count

    cdef int board_size
    cdef int group_penalty
    cdef int min_bet
    cdef int max_bet

    cdef State state
    cdef int bets[2]
    cdef int chosen_player
    cdef np.ndarray board
    cdef vector[Position] empty_positions
    cdef vector[Move] move_history


    cpdef int move_count(self) except *

    cpdef Move get_move(self, int index) except *

    cpdef void apply_move(self, int index) except *

    cpdef void undo_move(self) except *

    cpdef bool is_final(self) except *

    cpdef int count_groups(self, int player) except *

    cpdef np.ndarray[np.float32_t, ndim=1] payoff(self)

    cpdef np.ndarray[np.float32_t, ndim=2] as_matrix(self)

    cpdef np.ndarray[np.float32_t, ndim=1] random_playout_payoff(self)

    cpdef GameState __deepcopy__(self, memodict)
