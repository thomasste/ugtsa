from computation_graphs.computation_graph.computation_graph import \
    ComputationGraph

import tensorflow as tf
import numpy as np
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

    def set_player_count(self, player_count):
        self.player_count = player_count

    def set_worker_count(self, worker_count):
        self.worker_count = worker_count

    def set_statistic_size(self, statistic_size):
        self.statistic_size = statistic_size

    def set_update_size(self, update_size):
        self.update_size = update_size

    def set_game_state_board_shape(self, game_state_board_shape):
        self.game_state_board_shape = game_state_board_shape

    def set_game_state_statistic_size(self, game_state_statistic_size):
        self.game_state_statistic_size = game_state_statistic_size

    def set_update_statistic_size(self, update_statistic_size):
        self.update_statistic_size = update_statistic_size

    def set_seed_size(self, seed_size):
        self.seed_size = seed_size

    def _empty_statistic(self, training, global_step, seed, game_state_board,
                         game_state_statistic):
        raise NotImplementedError

    def _move_rate(self, training, global_step, seed, parent_statistic,
                   child_statistic):
        raise NotImplementedError

    def _game_state_as_update(self, training, global_step, seed,
                              update_statistic):
        raise NotImplementedError

    def _updated_statistic(self, training, global_step, seed, statistic,
                           update_count, updates):
        raise NotImplementedError

    def _updated_update(self, training, global_step, seed, update, statistic):
        raise NotImplementedError

    def _cost_function(self, training, global_step, move_rate, ucb_move_rate,
                       ugtsa_move_rate, trainable_variables):
        raise NotImplementedError

    def _apply_gradients(self, training, global_step, grads_and_vars):
        raise NotImplementedError

    @staticmethod
    def __model_gradients(variable_scope: tf.VariableScope,
                          transformation_variable_scope: tf.VariableScope,
                          output: tf.Tensor, output_gradient: tf.Tensor):
        trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            transformation_variable_scope.name)

        gradients = tf.gradients(output, trainable_variables, output_gradient)

        for gradient in gradients:
            gradient_accumulator = tf.Variable(tf.zeros(
                gradient.get_shape(), gradient.dtype))

            tf.add_to_collection(
                '{}/model_gradients'.format(variable_scope.name),
                gradient)
            tf.add_to_collection(
                '{}/model_gradient_accumulators'.format(variable_scope.name),
                gradient_accumulator)
            tf.add_to_collection(
                '{}/update_model_gradient_accumulators'.format(
                    variable_scope.name),
                tf.assign_add(gradient_accumulator, gradient).op)

        with tf.control_dependencies(tf.get_collection(
                "{}/update_model_gradient_accumulators".format(variable_scope.name))):
            # there is no noop
            tf.add(1, 1, "update_model_gradient_accumulators")

        tf.variables_initializer(
            tf.get_collection(
                '{}/model_gradient_accumulators'.format(variable_scope.name)),
            'zero_model_gradient_accumulators')

    def __build_empty_statistic_graph(self, training, global_step):
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
                signal = self._empty_statistic(
                    training, global_step,
                    seed, game_state_board, game_state_statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'output_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

    def __build_move_rate_graph(self, training, global_step):
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
                signal = self._move_rate(
                    training, global_step,
                    seed, parent_statistic, child_statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'output_gradient')

            parent_statistic_gradient, child_statistic_gradient = \
                tf.gradients(
                    output, [parent_statistic, child_statistic],
                    output_gradient)
            tf.identity(
                parent_statistic_gradient, 'parent_statistic_gradient')
            tf.identity(
                child_statistic_gradient, 'child_statistic_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

    def __build_game_state_as_update_graph(self, training, global_step):
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
                signal = self._game_state_as_update(
                    training, global_step,
                    seed, update_statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.update_size],
                'output_gradient')

            update_statistic_gradient, = tf.gradients(
                output, [update_statistic], output_gradient)
            tf.identity(
                update_statistic_gradient, 'update_statistic_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

    def __build_updated_statistic_graph(self, training, global_step):
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
                signal = self._updated_statistic(
                    training, global_step,
                    seed, statistic, update_count, updates)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.statistic_size],
                'output_gradient')

            statistic_gradient, updates_gradient = tf.gradients(
                output, [statistic, updates], output_gradient)
            tf.identity(statistic_gradient, 'statistic_gradient')
            tf.identity(updates_gradient, 'updates_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

    def __build_updated_update_graph(self, training, global_step):
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
                signal = self._updated_update(
                    training, global_step,
                    seed, update, statistic)

            output = tf.identity(signal, 'output')

            output_gradient = tf.placeholder(
                tf.float32,
                [None, self.update_size],
                'output_gradient')

            statistic_gradient, update_gradient = tf.gradients(
                output, [statistic, update], output_gradient)
            tf.identity(statistic_gradient, 'statistic_gradient')
            tf.identity(update_gradient, 'update_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output,
                output_gradient)

    def __build_cost_function_graph(self, training, global_step):
        print('cost_function')
        with tf.variable_scope('cost_function') as variable_scope:
            move_rates = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'move_rates')
            ucb_move_rates = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'ucb_move_rates')
            ugtsa_move_rates = tf.placeholder(
                tf.float32,
                [None, self.player_count],
                'ugtsa_move_rates')

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
                signal = self._cost_function(
                    training, global_step,
                    move_rates, ucb_move_rates, ugtsa_move_rates,
                    trainable_variables)

            output = tf.identity(signal, 'output')

            move_rates_gradient, = tf.gradients(output, [move_rates])
            tf.identity(move_rates_gradient, 'move_rates_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output, 1)

    def __build_apply_gradients_graph(self, training, global_step):
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

                model_gradient_accumulators = tf.get_collection(
                    '{}/model_gradient_accumulators'.format(name))

                trainable_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    '{}/transformation'.format(name))

                grads_and_vars += list(
                    zip(model_gradient_accumulators, trainable_variables))

            self._apply_gradients(training, global_step, grads_and_vars)

    def build(self):
        training = tf.placeholder(tf.bool, name='training')
        global_step = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int32,
            name='global_step')

        self.__build_empty_statistic_graph(training, global_step)
        self.__build_move_rate_graph(training, global_step)
        self.__build_game_state_as_update_graph(training, global_step)
        self.__build_updated_statistic_graph(training, global_step)
        self.__build_updated_update_graph(training, global_step)
        self.__build_cost_function_graph(training, global_step)
        self.__build_apply_gradients_graph(training, global_step)

    @classmethod
    def create_transformation(cls, computation_graph: ComputationGraph,
                              variable_scope_name: str, inputs: [(str, bool)])\
            -> ComputationGraph.Transformation:
        variable_template = '{}/{{}}:0'.format(variable_scope_name)
        gradient_template = '{}/{{}}_gradient:0'.format(variable_scope_name)
        operation_template = '{}/{{}}'.format(variable_scope_name)

        graph = tf.get_default_graph()

        return computation_graph.transformation(
            training=graph.get_tensor_by_name('training:0'),
            seed=graph.get_tensor_by_name(variable_template.format('seed')),
            inputs=[
                (graph.get_tensor_by_name(
                    variable_template.format(input_name)),
                 graph.get_tensor_by_name(
                    gradient_template.format(input_name))
                 if differentiable else None)
                for input_name, differentiable in inputs],
            output=(
                graph.get_tensor_by_name(variable_template.format('output')),
                graph.get_tensor_by_name(gradient_template.format('output'))),
            update_model_gradient_accumulators=graph.get_collection(
                operation_template.format(
                    'update_model_gradient_accumulators')))

    @classmethod
    def create_transformations(cls, computation_graph: ComputationGraph) \
            -> [ComputationGraph.Transformation]:
        return [
            cls.create_transformation(
                computation_graph, variable_scope_name, inputs)
            for variable_scope_name, inputs in [
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

    @classmethod
    def cost_function(cls, move_rates, ucb_move_rates, ugtsa_move_rates) \
            -> (float, [np.ndarray]):
        graph = tf.get_default_graph()

        return tf.get_default_session().run(
            [graph.get_tensor_by_name('cost_function/output:0'),
             graph.get_tensor_by_name('cost_function/move_rates_gradient:0')],
            {
                graph.get_tensor_by_name('cost_function/move_rates:0'):
                    move_rates,
                graph.get_tensor_by_name('cost_function/ucb_move_rates:0'):
                    ucb_move_rates,
                graph.get_tensor_by_name('cost_function/ugtsa_move_rates:0'):
                    ugtsa_move_rates
            })

    @classmethod
    def zero_model_gradient_accumulators(cls) -> None:
        zero_operations = [
            tf.get_default_graph().get_operation_by_name(
                '{}/zero_model_gradient_accumulators'.format(
                    variable_scope_name))
            for variable_scope_name in [
                'empty_statistic',
                'move_rate',
                'game_state_as_update',
                'updated_statistic',
                'updated_update',
                'cost_function']]

        tf.get_default_session().run(zero_operations)

    @classmethod
    def model_gradient_accumulators_debug_info(cls):
        session = tf.get_default_session()

        for name in [
                'empty_statistic',
                'move_rate',
                'game_state_as_update',
                'updated_statistic',
                'updated_update',
                'cost_function']:
            # trainable_variables = tf.get_collection(
            #     tf.GraphKeys.TRAINABLE_VARIABLES,
            #     '{}/transformation'.format(name))
            model_gradient_accumulators = tf.get_default_graph() \
                .get_collection('{}/model_gradient_accumulators'.format(name))

            #print(session.run(trainable_variables[:2]))
            print(tf.get_default_graph().get_collection('{}/model_gradient_accumulators'.format(name)))
            print(session.run(model_gradient_accumulators[:]))

    @classmethod
    def apply_gradients(cls) -> None:
        tf.get_default_session().run(
            tf.get_default_graph().get_operation_by_name(
                'apply_gradients/apply_gradients'))
