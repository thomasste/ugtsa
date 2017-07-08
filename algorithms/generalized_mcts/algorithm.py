from algorithms.algorithm import algorithm
from copy import deepcopy
from enum import Enum
from games.game.game_state import GameState
from itertools import groupby
from numpy.random import choice
from recordclass import recordclass

import numpy as np


class Algorithm(algorithm.Algorithm):
    class Direction(Enum):
        UP = 0
        DOWN = 1

    Worker = recordclass('Worker', 'node direction game_state')
    Node = recordclass('Node', 'number_of_visits move statistic parent children')
    Statistic = object

    def __init__(self, game_state: GameState, number_of_workers: int, grow_factor: int):
        super(Algorithm, self).__init__(game_state)

        self.workers = [
            Algorithm.Worker(
                node=0,
                direction=Algorithm.Direction.DOWN,
                game_state=deepcopy(game_state))
            for _ in range(number_of_workers)]
        self.tree = [
            Algorithm.Node(
                number_of_visits=0,
                move=None,
                statistic=self._empty_statistic([GameState])[0],
                parent=None,
                children=None)]
        self.grow_factor = grow_factor

    # O(tree_size)
    def _empty_statistic(self, game_state: [GameState]) -> [Statistic]:
        raise NotImplementedError

    # O(number_of_workers * tree_height * tree_ramification * number_of_iterations)
    # TODO: remove tree_ramification factor (add cache)
    def _move_rate(self, parent_statistic: [Statistic], child_statistic: [Statistic]) -> [np.matrix]:
        raise NotImplementedError

    # O(number_of_iterations * number_of_workers)
    def _updated_leaf_statistic(self, statistic: [Statistic], payoffs: [[np.matrix]]) -> [Statistic]:
        raise NotImplementedError

    # O(number_of_workers * tree_height * number_of_iterations)
    def _updated_node_statistic(self, parent_statistic: [Statistic], children_statistics: [[Statistic]]) -> [Statistic]:
        raise NotImplementedError

    # O(number_of_workers * number_of_iterations)
    def _random_playout_payoff(self, game_state: GameState) -> np.matrix:
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

    def __down_payoff_case(self, workers, statistic):
        node = self.tree[workers[0].node]

        node.statistic = statistic

        for worker in workers:
            worker.direction = Algorithm.Direction.UP

    def __up_non_root_case(self, workers, statistic):
        pidx = self.tree[workers[0].node].parent

        self.tree[pidx].statistic = statistic

        for worker in workers:
            self.tree[worker.node].number_of_visits += 1
            worker.game_state.undo_move(self.tree[worker.node].move)
            worker.node = pidx

    def __up_root_case(self, workers):
        self.tree[0].number_of_visits += len(workers)

        for worker in workers:
            worker.direction = Algorithm.Direction.DOWN

    def improve(self):
        empty_statistics_0 = []
        move_rate_0 = []
        move_rate_1 = []
        updated_leaf_statistics_0 = []
        updated_leaf_statistics_1 = []
        updated_node_statistics_0 = []
        updated_node_statistics_1 = []

        up_workers = groupby(sorted(filter(
            lambda x: x.direction == Algorithm.Direction.UP, self.workers),
            key=lambda x: x.node), key=lambda x: x.node)
        down_workers = groupby(sorted(filter(
            lambda x: x.direction == Algorithm.Direction.DOWN, self.workers),
            key=lambda x: x.parent), key=lambda x: x.parent)

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
                        empty_statistics_0 += [deepcopy(workers[0].game_state).apply_move(move)]
                else:
                    updated_leaf_statistics_0 += [node.statistics]
                    updated_leaf_statistics_1 += [[self._random_playout_payoff(worker.game_state)
                                                   for worker in workers]]

        # UP
        for pidx, workers in up_workers:
            if pidx:
                parent_node = self.tree[pidx]

                updated_node_statistics_0 += [parent_node.statistics]
                updated_node_statistics_1 += [[self.tree[idx].statistic for idx, _ in groupby(workers, key=lambda x: x.node)]]
            else:
                pass

        empty_statistics = self._empty_statistic(empty_statistics_0)
        move_rate = self._move_rate(move_rate_0, move_rate_1)
        updated_leaf_statistics = self._updated_leaf_statistic(updated_leaf_statistics_0, updated_leaf_statistics_1)
        updated_node_statistics = self._updated_node_statistic(updated_node_statistics_0, updated_node_statistics_1)

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
                    self.__down_payoff_case(workers, result)
                else:
                    result, updated_leaf_statistics = updated_leaf_statistics[0], updated_leaf_statistics[1:]
                    self.__down_payoff_case(workers, result)

        # UP
        for pidx, workers in up_workers:
            if pidx:
                result, updated_node_statistics = updated_node_statistics[0], updated_node_statistics[1:]
                self.__up_non_root_case(workers, result)
            else:
                self.__up_root_case(workers)
