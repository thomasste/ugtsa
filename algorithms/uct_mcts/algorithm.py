from algorithms.generalized_mcts import algorithm
from games.game.game_state import GameState
from recordclass import recordclass

import numpy as np


class Algorithm(algorithm.Algorithm):
    Statistic = recordclass('Statistic', 'w n')

    def __init__(self, game_state: GameState, number_of_workers: int, grow_factor: int, exploration_factor: int):
        super(Algorithm, self).__init__(game_state, number_of_workers, grow_factor)
        self.exploration_factor = exploration_factor

    def _empty_statistic(self, game_state: [GameState]):
        return [Algorithm.Statistic(w=np.zeros(game_state[0].player_count), n=0)
                for _ in game_state]

    def _move_rate(self, parent_statistic: [Statistic], child_statistic: [Statistic]):
        return [cs.w / (cs.n + 0.1) + self.exploration_factor * np.sqrt(np.log(ps.n) / (cs.n + 0.1))
                for (ps, cs) in zip(parent_statistic, child_statistic)]

    def _updated_statistic(self, statistic: [Statistic], payoffs: [[np.array]]) -> [Statistic]:
        result = []

        for s, ps in statistic, payoffs:
            ns = Algorithm.Statistic(w=np.array(s.w, copy=True), n=s.n + len(ps))
            for w in np.argmax(ps, 1):
                ns.w[w] += 1
            result += [ns]

        return result

    # inne opcje:
    # - za każdym razem zwijam statystyki dzieci - nie umiem w tym wyrazić mcts, dużo liczenia
    # - za każdym razem jak idę w górę to przekazuję statystyki dziecka i payoff - ok
    # - zmieniam payoff w "coś" i to coś rekurencyjnie updatuję w górę drzewa za pomocą statystyk - ok
