from algorithms.algorithm import algorithm
from copy import deepcopy
from enum import Enum
from games.game.game_state import GameState
from itertools import groupby
from numpy.random import choice
from recordclass import recordclass


class Algorithm(algorithm.Algorithm):
    class Direction(Enum):
        UP = 0
        DOWN = 1

    Worker = recordclass(
        'Worker', 'node direction game_state update')
    Node = recordclass(
        'Node', 'number_of_visits statistic parent children '
                'move_rate_cache')

    Statistic = object
    Update = object
    Rate = object

    def __init__(self, game_state: GameState, worker_count: int,
                 grow_factor: int, removed_root_moves: [int]):
        super().__init__(game_state)

        self.workers = [
            Algorithm.Worker(
                node=0,
                direction=Algorithm.Direction.DOWN,
                game_state=deepcopy(game_state),
                update=None)
            for _ in range(worker_count)]
        self.tree = []
        self.grow_factor = grow_factor
        self.removed_root_moves = removed_root_moves

    def _empty_statistic(self, game_state: GameState) -> Statistic:
        raise NotImplementedError

    def _move_rate(
            self, parent_statistic: Statistic, child_statistic: Statistic) \
            -> Rate:
        raise NotImplementedError

    def _game_state_as_update(self, game_state: GameState) -> Update:
        raise NotImplementedError

    def _updated_statistic(self, statistic: Statistic, updates: [Update]) \
            -> Statistic:
        raise NotImplementedError

    def _updated_update(self, update: Update, statistic: Statistic) \
            -> Update:
        raise NotImplementedError

    def _run_batch(self) -> None:
        raise NotImplementedError

    def __down_move_case(self, workers):
        node = self.tree[workers[0].node]

        if workers[0].game_state.player != -1:
            move_rates = [
                self.value(self.tree[cidx].move_rate_cache)
                for cidx in range(node.children[0], node.children[1])]

            # make probabilities
            xs = [x[workers[0].game_state.player] for x in move_rates]

            # remove moves
            if node.parent is None:
                for removed_root_move in self.removed_root_moves:
                    xs[removed_root_move] = 0.

            xs = [x / sum(xs) for x in xs]
        else:
            xs = [1 / (node.children[1] - node.children[0])
                  for _ in range(node.children[1] - node.children[0])]

        cidxs = choice(
            range(node.children[0], node.children[1]), len(workers), p=xs)

        for cidx, worker in zip(cidxs, workers):
            worker.node = cidx
            worker.game_state.apply_move(cidx - node.children[0])

    def __down_expand_case(self, workers, statistics):
        node = self.tree[workers[0].node]

        node.children = (len(self.tree), len(self.tree) + len(statistics))

        for statistic in statistics:
            self.tree += [Algorithm.Node(
                number_of_visits=0,
                statistic=statistic,
                parent=workers[0].node,
                children=None,
                move_rate_cache=None)]

    def __down_leaf_case(self, workers, updates):
        for worker, update in zip(workers, updates):
            worker.direction = Algorithm.Direction.UP
            worker.update = update

    def __up_case(self, workers, statistic, updates):
        node = self.tree[workers[0].node]

        node.move_rate_cache = None
        node.statistic = statistic
        node.number_of_visits += len(workers)

        for worker, update in zip(workers, updates):
            worker.update = update

        if node.parent is not None:
            for worker in workers:
                worker.game_state.undo_move()
                worker.node = node.parent
        else:
            for worker in workers:
                worker.update = None
                worker.direction = Algorithm.Direction.DOWN

    def improve(self):
        if not self.tree:
            self.tree += [
                Algorithm.Node(
                    number_of_visits=0,
                    statistic=self._empty_statistic(
                        self.workers[0].game_state),
                    parent=None,
                    children=None,
                    move_rate_cache=None)]
            self._run_batch()

        empty_statistic = []
        move_rate = []
        game_state_as_update = []
        updated_statistic = []
        updated_update = []

        up_workers = groupby(sorted(filter(
            lambda x: x.direction == Algorithm.Direction.UP, self.workers),
            key=lambda x: x.node), key=lambda x: x.node)
        up_workers = [(idx, list(workers)) for idx, workers in up_workers]

        down_workers = groupby(sorted(filter(
            lambda x: x.direction == Algorithm.Direction.DOWN, self.workers),
            key=lambda x: x.node), key=lambda x: x.node)
        down_workers = [(idx, list(workers)) for idx, workers in down_workers]

        # DOWN
        for idx, workers in down_workers:
            node = self.tree[idx]

            if node.children is not None:
                for cidx in range(node.children[0], node.children[1]):
                    if self.tree[cidx].move_rate_cache is None:
                        move_rate += [self._move_rate(
                            node.statistic, self.tree[cidx].statistic)]
            else:
                if node.number_of_visits >= self.grow_factor \
                        and not workers[0].game_state.is_final():
                    for move_index in \
                            range(workers[0].game_state.move_count()):
                        workers[0].game_state.apply_move(move_index)
                        empty_statistic += [self._empty_statistic(
                            workers[0].game_state)]
                        workers[0].game_state.undo_move()
                else:
                    for worker in workers:
                        game_state_as_update += [self._game_state_as_update(
                            worker.game_state)]

        # UP
        for idx, workers in up_workers:
            node = self.tree[idx]

            updated_statistic += [self._updated_statistic(
                node.statistic, [worker.update for worker in workers])]

            for worker in workers:
                updated_update += [self._updated_update(
                    worker.update, node.statistic)]

        self._run_batch()

        # DOWN
        for idx, workers in down_workers:
            node = self.tree[idx]

            if node.children is not None:
                # fill gaps
                for cidx in range(node.children[0], node.children[1]):
                    if self.tree[cidx].move_rate_cache is None:
                        self.tree[cidx].move_rate_cache, move_rate = \
                            move_rate[0], move_rate[1:]

                self.__down_move_case(workers)
            else:
                if node.number_of_visits >= self.grow_factor \
                        and not workers[0].game_state.is_final():
                    move_count = workers[0].game_state.move_count()
                    result, empty_statistic = \
                        empty_statistic[:move_count], \
                        empty_statistic[move_count:]
                    self.__down_expand_case(workers, result)
                else:
                    result, game_state_as_update = \
                        game_state_as_update[:len(workers)], \
                        game_state_as_update[len(workers):]
                    self.__down_leaf_case(workers, result)

        # UP
        for idx, workers in up_workers:
            result1, updated_statistic = \
                updated_statistic[0], updated_statistic[1:]
            result2, updated_update = \
                updated_update[:len(workers)], updated_update[len(workers):]

            self.__up_case(workers, result1, result2)

    def move_rates(self) -> [Rate]:
        root = self.tree[0]
        assert root.children

        result = []
        for idx in range(self.tree[0].children[0], self.tree[0].children[1]):
            node = self.tree[idx]
            result += [self._move_rate(root.statistic, node.statistic)]
        self._run_batch()
        return result
