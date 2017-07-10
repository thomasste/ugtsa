from algorithms.algorithm import algorithm
from copy import deepcopy
from enum import Enum
from games.game.game_state import GameState
from games.game.move import Move
from itertools import groupby
from numpy.random import choice
from recordclass import recordclass

import numpy as np


class Algorithm(algorithm.Algorithm):
    class Direction(Enum):
        UP = 0
        DOWN = 1

    Worker = recordclass('Worker', 'node direction game_state payoff')
    Node = recordclass('Node', 'number_of_visits move statistic parent children')
    Statistic = object

    def __init__(self, game_state: GameState, number_of_workers: int, grow_factor: int):
        super(Algorithm, self).__init__(game_state)

        self.workers = [
            Algorithm.Worker(
                node=0,
                direction=Algorithm.Direction.DOWN,
                game_state=deepcopy(game_state),
                payoff=None)
            for _ in range(number_of_workers)]
        self.tree = [
            Algorithm.Node(
                number_of_visits=0,
                move=None,
                statistic=self._empty_statistic([GameState])[0],
                parent=None,
                children=None)]
        self.grow_factor = grow_factor

    def _empty_statistic(self, game_state: [GameState]) -> [Statistic]:
        raise NotImplementedError

    def _move_rate(self, parent_statistic: [Statistic], child_statistic: [Statistic]) -> [np.array]:
        raise NotImplementedError

    def _updated_statistic(self, statistic: [Statistic], payoffs: [[np.array]]) -> [Statistic]:
        raise NotImplementedError

    def _random_playout_payoff(self, game_state: GameState) -> np.array:
        raise NotImplementedError

    def __down_children_case(self, workers, move_rates):
        # TODO: optimization - cache move_rate in tree
        node = self.tree[workers[0].node]

        xs = [x[workers[0].game_state.player] for x in move_rates]
        m = min(xs)
        xs = [x - m for x in xs]
        s = sum(xs)
        xs = [x / s for x in xs]

        cidxs = choice(range(node.children[0], node.children[1]), len(workers), p=xs)

        for cidx, worker in zip(cidxs, workers):
            worker.node = cidx
            worker.game_state.apply_move(self.tree[worker.node].move)

    def __down_expand_case(self, workers, statistics):
        node = self.tree[workers[0].node]

        node.children = (len(self.tree), len(self.tree) + len(statistics))

        for statistic, move in zip(statistics, workers[0].game_state.moves()):
            self.tree += [Algorithm.Node(
                number_of_visits=0,
                move=move,
                statistic=statistic,
                parent=workers[0].node,
                children=None)]

    def __down_payoff_case(self, workers, payoffs):
        for worker, payoff in zip(workers, payoffs):
            worker.direction = Algorithm.Direction.UP
            worker.payoff = payoff

    def __up_case(self, workers, statistic):
        node = self.tree[workers[0].node]

        node.statistic = statistic
        node.number_of_visits += len(workers)

        if node.parent:
            for worker in workers:
                worker.game_state.undo_move(node.move)
                worker.node = node.parent
        else:
            for worker in workers:
                worker.payoff = None
                worker.direction = Algorithm.Direction.DOWN

    def improve(self):
        empty_statistic_0 = []
        move_rate_0 = []
        move_rate_1 = []
        updated_statistic_0 = []
        updated_statistic_1 = []

        up_workers = groupby(sorted(filter(
            lambda x: x.direction == Algorithm.Direction.UP, self.workers),
            key=lambda x: x.node), key=lambda x: x.node)
        down_workers = groupby(sorted(filter(
            lambda x: x.direction == Algorithm.Direction.DOWN, self.workers),
            key=lambda x: x.node), key=lambda x: x.node)

        # DOWN
        for idx, workers in down_workers:
            node = self.tree[idx]

            if node.children:
                for cidx in range(node.children[0], node.children[1]):
                    move_rate_0 += [node.statistics]
                    move_rate_1 += [self.tree[cidx].statistics]
            else:
                if node.number_of_visits >= self.grow_factor:
                    for move in workers[0].game_state.moves():
                        empty_statistic_0 += [deepcopy(workers[0].game_state).apply_move(move)]
                else:
                    pass

        # UP
        for idx, workers in up_workers:
            node = self.tree[idx]

            updated_statistic_0 += [node.statistic]
            updated_statistic_1 += [worker.payoff for worker in workers]

        empty_statistics = self._empty_statistic(empty_statistic_0)
        move_rate = self._move_rate(move_rate_0, move_rate_1)
        updated_statistic = self._updated_statistic(updated_statistic_0, updated_statistic_1)

        # DOWN
        for idx, workers in down_workers:
            node = self.tree[idx]

            if node.children:
                move_count = node.children[1] - node.children[0]
                result, move_rate = move_rate[:move_count], move_rate[move_count:]
                self.__down_children_case(workers, result)
            else:
                if node.number_of_visits >= self.grow_factor:
                    move_count = len(workers[0].game_state.moves())
                    result, empty_statistics = empty_statistics[move_count]
                    self.__down_expand_case(workers, result)
                else:
                    self.__down_payoff_case(
                        workers, [[self._random_playout_payoff(worker.game_state) for worker in workers]])

        # UP
        for idx, workers in up_workers:
            result, updated_statistic = updated_statistic[0], updated_statistic[1:]
            self.__up_case(workers, result)

    def move_rates(self) -> [(Move, np.array)]:
        root = self.tree[0]

        assert root.children

        result = []
        for idx in range(self.tree[0].children[0], self.tree[0].children[1]):
            node = self.tree[idx]
            result += [(node.move, self._move_rate([root.statistic], [node.statistic])[0])]
        return result
