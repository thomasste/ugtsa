from games.omringa.game_state import GameState
from algorithms.ucb_mcts.algorithm import Algorithm
import numpy as np


def human_vs_human_playout(game_state):
    stack = []

    while not game_state.is_final():
        print(game_state)
        try:
            print(game_state.payoff())
        except:
            pass

        for i, move in enumerate(game_state.moves()):
            print('{}: {}'.format(i, move))

        index = int(input("Move: "))
        # index = 0

        if index == -1:
            move, stack = stack[-1], stack[:-1]
            game_state.undo_move(move)
        else:
            stack += [game_state.moves()[index]]
            game_state.apply_move(stack[-1])

    print(game_state)
    print(game_state.payoff())


def human_vs_ai_playout(game_state, human=0):
    while not game_state.is_final():
        print(game_state)
        try:
            print(game_state.payoff())
        except:
            pass

        for i, move in enumerate(game_state.moves()):
            print('{}: {}'.format(i, move))

        if game_state.player == human:
            # index = int(input("Move: "))
            index = 0
            game_state.apply_move(game_state.moves()[index])
        else:
            algorithm = Algorithm(game_state, 1, 5, np.sqrt(2))
            for i in range(1000):
                algorithm.improve()
            print(algorithm.move_rates())
            print(algorithm.best_move())
            move = algorithm.best_move()
            game_state.apply_move(move)

    print(game_state)
    print(game_state.payoff())

game_state = GameState(7)
# human_vs_human_playout(game_state)
human_vs_ai_playout(game_state)