from computation_graphs.computation_graph.computation_graph import \
    ComputationGraph

import tensorflow as tf
import itertools


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

    def _cost_function_transformation(
            self, move_rate, ucb_move_rate, ugtsa_move_rate,
            trainable_variables):
        raise NotImplementedError

    def _apply_gradients(self, grads_and_vars):
        raise NotImplementedError

    def __model_gradients(
            self, variable_scope: tf.VariableScope,
            transformation_variable_scope: tf.VariableScope,
            output: tf.Tensor, output_gradient: tf.Tensor):
        trainable_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                transformation_variable_scope.name)

        gradients = tf.gradients(
            output, trainable_variables, grad_ys=output_gradient)

        for gradient in gradients:
            gradient_accumulator = tf.Variable(
                tf.zeros(gradient.get_shape(), gradient.dtype))

            tf.add_to_collection(
                '{}/model_gradients'.format(variable_scope.name),
                gradient)
            tf.add_to_collection(
                '{}/model_gradient_accumulators'.format(variable_scope.name),
                gradient_accumulator)
            tf.add_to_collection(
                '{}/update_model_gradient_accumulators'.format(
                    variable_scope.name),
                tf.assign_add(
                    gradient_accumulator, gradient, use_locking=True).op)

        tf.variables_initializer(
            tf.get_collection(
                '{}/model_gradient_accumulators'.format(variable_scope.name)),
            'zero_model_gradient_accumulators')

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

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

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

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

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

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

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

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

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

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

    def __build_cost_function_graph(self):
        print('cost_function')
        with tf.variable_scope('cost_function') as variable_scope:
            move_rate = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'move_rate')
            ucb_move_rate = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'ucb_move_rate')
            ugtsa_move_rate = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'ugtsa_move_rate')

            trainable_variables = itertools.chain(*[
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    '{}/transformation'.format(name))
                for name in [
                    'empty_statistic',
                    'move_rate',
                    'game_state_as_update',
                    'updated_statistic',
                    'updated_update']])

            with tf.variable_scope('transformation') as \
                    transformation_variable_scope:
                signal = self._cost_function_transformation(
                    move_rate, ucb_move_rate, ugtsa_move_rate,
                    trainable_variables)

            output = tf.identity(signal, 'output')

            move_rate_gradient, = tf.gradients(output, [move_rate])
            tf.identity(move_rate_gradient, 'move_rate_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output, 1)

    def __build_apply_gradients_graph(self):
        print('apply_gradients')
        with tf.variable_scope('apply_gradients'):
            grads_and_vars = []
            for name in [
                    'empty_statistic',
                    'move_rate',
                    'game_state_as_update',
                    'updated_statistic',
                    'updated_update',
                    'cost_function']:
                trainable_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    '{}/transformation'.format(name))
                model_gradient_accumulators = tf.get_default_graph()\
                    .get_collection('{}/model_gradient_accumulators'.format(
                        name))
                for variable, gradient in zip(
                        trainable_variables, model_gradient_accumulators):
                    grads_and_vars += [(gradient, variable)]

            self._apply_gradients(grads_and_vars)

    def build(self):
        with tf.variable_scope('settings'):
            self.training = tf.placeholder(tf.bool, name='training')

        with tf.variable_scope('globals'):
            self.global_step = tf.Variable(
                initial_value=0, dtype=tf.int32, name='global_step',
                trainable=False)

        self.__build_empty_statistic_graph()
        self.__build_move_rate_graph()
        self.__build_game_state_as_update_graph()
        self.__build_updated_statistic_graph()
        self.__build_updated_update_graph()
        self.__build_cost_function_graph()
        self.__build_apply_gradients_graph()

    @classmethod
    def transformation(
            cls, computation_graph: ComputationGraph, name: str,
            inputs: [(str, bool)]) -> ComputationGraph.Transformation:
        template = '{}/{{}}'.format(name)
        template_0 = '{}/{{}}:0'.format(name)
        template_g0 = '{}/{{}}_gradient:0'.format(name)

        graph = tf.get_default_graph()

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
            update_model_gradient_accumulators=graph.get_collection(
                template.format('update_model_gradient_accumulators')),
            seed=graph.get_tensor_by_name(template_0.format('seed')),
            training=graph.get_tensor_by_name('settings/training:0'))

    @classmethod
    def transformations(
            cls, computation_graph: ComputationGraph) \
            -> [ComputationGraph.Transformation]:
        return [
            cls.transformation(computation_graph, name, inputs)
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
