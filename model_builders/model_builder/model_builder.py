from computation_graphs.computation_graph.computation_graph import \
    ComputationGraph

import tensorflow as tf


class ModelBuilder(object):
    def __init__(
            self, player_count, worker_count, statistic_size, update_size,
            game_state_board_shape, game_state_statistic_size,
            update_statistic_size, seed_size):
        self.player_count = player_count
        self.worker_count = worker_count
        self.statistic_size = statistic_size
        self.update_size = update_size
        self.game_state_board_shape = game_state_board_shape
        self.game_state_statistic_size = game_state_statistic_size
        self.update_statistic_size = update_statistic_size
        self.seed_size = seed_size

    def _empty_statistic_transformation(
            self, seed, game_state_board, game_state_statistic):
        raise NotImplementedError

    def _move_rate_transformation(
            self, seed, parent_statistic, child_statistic):
        raise NotImplementedError

    def _game_state_as_update_transformation(self, seed, update_statistic):
        raise NotImplementedError

    def _updated_statistic_transformation(
            self, seed, statistic, update_count, updates):
        raise NotImplementedError

    def _updated_update_transformation(self, seed, update, statistic):
        raise NotImplementedError

    def __build_empty_statistic_graph(self):
        print('empty_statistic')
        with tf.variable_scope('empty_statistic') as variable_scope:
            seed = tf.placeholder(
                tf.int64,
                [self.seed_size],
                'seed')
            game_state_board = tf.placeholder(
                tf.float32,
                [None,
                 self.game_state_board_shape[0],
                 self.game_state_board_shape[1]],
                'game_state_board')
            game_state_statistic = tf.placeholder(
                tf.float32,
                [None, self.game_state_statistic_size],
                'game_state_statistic')

            with tf.variable_scope('transformation') as \
                    transformation_variable_scope:
                signal = self._empty_statistic_transformation(
                    seed, game_state_board, game_state_statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'output_gradient')

            model_gradients = tf.gradients(
                output, tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    transformation_variable_scope.name),
                grad_ys=output_gradient)

            for gradient in model_gradients:
                tf.add_to_collection(
                    '{}/model_gradients'.format(variable_scope.name),
                    gradient)

    def __build_move_rate_graph(self):
        print('move_rate')
        with tf.variable_scope('move_rate') as variable_scope:
            seed = tf.placeholder(
                tf.int64,
                [self.seed_size],
                'seed')
            parent_statistic = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'parent_statistic')
            child_statistic = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'child_statistic')

            with tf.variable_scope('transformation') as \
                    transformation_variable_scope:
                signal = self._move_rate_transformation(
                    seed, parent_statistic, child_statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'output_gradient')

            parent_statistic_gradient, child_statistic_gradient = \
                tf.gradients(
                    output, [parent_statistic, child_statistic],
                    grad_ys=output_gradient)
            tf.identity(
                parent_statistic_gradient, 'parent_statistic_gradient')
            tf.identity(
                child_statistic_gradient, 'child_statistic_gradient')

            model_gradients = tf.gradients(
                output, tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    transformation_variable_scope.name),
                grad_ys=output_gradient)

            for gradient in model_gradients:
                tf.add_to_collection(
                    '{}/model_gradients'.format(variable_scope.name),
                    gradient)

    def __build_game_state_as_update_graph(self):
        print('game_state_as_update')
        with tf.variable_scope('game_state_as_update') as variable_scope:
            seed = tf.placeholder(
                tf.int64,
                [self.seed_size],
                'seed')
            update_statistic = tf.placeholder(
                tf.float32,
                [None, self.update_statistic_size],
                'update_statistic')

            with tf.variable_scope('transformation') as \
                    transformation_variable_scope:
                signal = self._game_state_as_update_transformation(
                    seed, update_statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.update_size],
                'output_gradient')

            update_statistic_gradient = tf.gradients(
                output, update_statistic, grad_ys=output_gradient)
            tf.identity(
                update_statistic_gradient, 'update_statistic_gradient')

            model_gradients = tf.gradients(
                output, tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    transformation_variable_scope.name),
                grad_ys=output_gradient)

            for gradient in model_gradients:
                tf.add_to_collection(
                    '{}/model_gradients'.format(variable_scope.name),
                    gradient)

    def __build_updated_statistic_graph(self):
        print('updated_statistic')
        with tf.variable_scope('updated_statistic') as variable_scope:
            seed = tf.placeholder(
                tf.int64,
                [self.seed_size],
                'seed')
            statistic = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'statistic')
            update_count = tf.placeholder(
                tf.int32,
                [None],
                'update_count')
            updates = tf.placeholder(
                tf.float32,
                [None, self.update_size * self.worker_count],
                'updates')

            with tf.variable_scope('transformation') as \
                    transformation_variable_scope:
                signal = self._updated_statistic_transformation(
                    seed, statistic, update_count, updates)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'output_gradient')

            statistic_gradient, updates_gradient = tf.gradients(
                output, [statistic, updates], grad_ys=output_gradient)
            tf.identity(statistic_gradient, 'statistic_gradient')
            tf.identity(updates_gradient, 'updates_gradient')

            model_gradients = tf.gradients(
                output, tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    transformation_variable_scope.name),
                grad_ys=output_gradient)

            for gradient in model_gradients:
                tf.add_to_collection(
                    '{}/model_gradients'.format(variable_scope.name),
                    gradient)

    def __build_updated_update_graph(self):
        print('updated_update')
        with tf.variable_scope('updated_update') as variable_scope:
            seed = tf.placeholder(
                tf.int64,
                [self.seed_size],
                'seed')
            update = tf.placeholder(
                tf.float32,
                [None, self.update_size],
                'update')
            statistic = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'statistic')

            with tf.variable_scope('transformation') as \
                    transformation_variable_scope:
                signal = self._updated_update_transformation(
                    seed, update, statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.update_size],
                'output_gradient')

            statistic_gradient, update_gradient = tf.gradients(
                output, [statistic, update], grad_ys=output_gradient)
            tf.identity(statistic_gradient, 'statistic_gradient')
            tf.identity(update_gradient, 'update_gradient')

            model_gradients = tf.gradients(
                output, tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    transformation_variable_scope.name),
                grad_ys=output_gradient)

            for gradient in model_gradients:
                tf.add_to_collection(
                    '{}/model_gradients'.format(variable_scope.name),
                    gradient)

    def build(self):
        with tf.variable_scope('settings'):
            self.training = tf.placeholder(tf.bool, name='training')

        self.__build_empty_statistic_graph()
        self.__build_move_rate_graph()
        self.__build_game_state_as_update_graph()
        self.__build_updated_statistic_graph()
        self.__build_updated_update_graph()

    @classmethod
    def transformation(
            cls, computation_graph: ComputationGraph, graph: tf.Graph,
            name: str, inputs: [(str, bool)]) \
            -> ComputationGraph.Transformation:
        template = '{}/{{}}'.format(name)
        template_0 = '{}/{{}}:0'.format(name)
        template_g0 = '{}/{{}}_gradient:0'.format(name)

        return computation_graph.transformation(
            inputs=[
                graph.get_tensor_by_name(template_0.format(input_name))
                for input_name, _ in inputs],
            input_gradients=[
                graph.get_tensor_by_name(template_g0.format(input_name))
                if is_differentiable else None
                for input_name, is_differentiable in inputs],
            output=graph.get_tensor_by_name(template_0.format('output')),
            output_gradient=graph.get_tensor_by_name(
                template_g0.format('output')),
            model_gradients=graph.get_collection(
                template.format('model_gradients')),
            seed=graph.get_tensor_by_name(template_0.format('seed')),
            training=graph.get_tensor_by_name('settings/training:0'))

    @classmethod
    def transformations(
            cls, computation_graph: ComputationGraph, graph: tf.Graph) \
            -> [ComputationGraph.Transformation]:
        return [
            cls.transformation(computation_graph, graph, name, inputs)
            for name, inputs in [
                ('empty_statistic', [
                    ('game_state_board', False),
                    ('game_state_statistic', False)]),
                ('move_rate', [
                    ('parent_statistic', True),
                    ('child_statistic', True)]),
                ('game_state_as_update', [
                    ('update_statistic', False)]),
                ('updated_statistic', [
                    ('statistic', True),
                    ('update_count', False),
                    ('updates', True)]),
                ('updated_update', [
                    ('update', True),
                    ('statistic', True)])]]
