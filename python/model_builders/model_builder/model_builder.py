from computation_graphs.computation_graph.computation_graph import \
    ComputationGraph

import tensorflow as tf
import numpy as np
import itertools


class ModelBuilder(object):
    def __init__(
            self, player_count, worker_count,
            statistic_size, update_size, game_state_board_shape,
            game_state_statistic_size, update_statistic_size, seed_size):
        self.player_count = player_count
        self.game_state_count = 10
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

    def _empty_statistic(self, training, seed, game_state_board,
                         game_state_statistic):
        raise NotImplementedError

    def _move_rate(self, training, seed, parent_statistic,
                   child_statistic):
        raise NotImplementedError

    def _game_state_as_update(self, training, seed, update_statistic):
        raise NotImplementedError

    def _updated_statistic(self, training, seed, statistic, update_count,
                           updates):
        raise NotImplementedError

    def _updated_update(self, training, seed, update, statistic):
        raise NotImplementedError

    def _cost_function(self, global_step, move_rate, ucb_move_rate,
                       ugtsa_move_rate, trainable_variables):
        raise NotImplementedError

    def _apply_gradients(self, global_step, grads_and_vars):
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

        tf.variables_initializer(
            tf.get_collection(
                '{}/model_gradient_accumulators'.format(variable_scope.name)),
            'zero_model_gradient_accumulators')

    def __build_function_graph(self, name, inputs, output, transformation):
        print(name)
        with tf.variable_scope(name) as variable_scope:
            placeholders = {
                'training': tf.placeholder(
                    tf.bool,
                    [None],
                    'training'),
                'index': tf.placeholder(
                    tf.int32,
                    [None],
                    'index'),
                'seed': tf.placeholder(
                    tf.int64,
                    [None, self.seed_size],
                    'seed'),
                'output_gradient': tf.placeholder(
                    tf.float32,
                    [None] + output['shape'],
                    'output_gradient'),
                'batch_size': tf.placeholder(
                    tf.int32,
                    [None],
                    'batch_size'),
                **{
                    input_name: tf.placeholder(
                        inputs[input_name]['type'],
                        [None] + inputs[input_name]['shape'],
                        input_name)
                    for input_name in inputs
                }
            }

            # FORWARD
            input_queue = tf.FIFOQueue(
                capacity=self.game_state_count * self.worker_count,
                dtypes=[tf.int32, tf.bool, tf.int64] + [
                    inputs[input_name]['type']
                    for input_name in inputs],
                shapes=[[], [], [self.seed_size]] + [
                    inputs[input_name]['shape']
                    for input_name in inputs],
                names=['index', 'training', 'seed'] + [
                    input_name
                    for input_name in inputs])

            output_queue = tf.FIFOQueue(
                capacity=self.game_state_count * self.worker_count,
                dtypes=[tf.int32, tf.int32, tf.float32],
                shapes=[[], [], self.statistic_size],
                names=['index', 'batch_index', 'output'])

            # create input
            input_queue.enqueue_many({
                'training': placeholders['training'],
                'index': placeholders['index'],
                'seed': placeholders['seed'],
                **{
                    input_name: placeholders[input_name]
                    for input_name in inputs
                }}, 'forward_enqueue')

            # create flow
            batch_size = input_queue.size()
            batch = input_queue.dequeue_many(batch_size)

            with tf.variable_scope('transformation'):
                signal = tf.cond(
                    tf.size(batch['index']) > 0,
                    lambda: transformation(**{
                        'training': batch['training'][0],
                        'seed': batch['seed'][0],
                        **{input_name: batch[input_name]
                           for input_name in inputs}}),
                    lambda: tf.zeros((0, self.statistic_size), tf.float32))

            batch_index = tf.cond(
                tf.size(batch['index']) > 0,
                lambda: tf.tile([batch['index'][0]], [batch_size]),
                lambda: tf.zeros((0,), tf.int32))

            enqueue_op = output_queue.enqueue_many({
                'index': batch['index'],
                'batch_index': batch_index,
                'output': signal
            })

            queue_runner = tf.train.QueueRunner(output_queue, enqueue_op)
            tf.train.add_queue_runner(queue_runner)

            # create output
            o = output_queue.dequeue_many(output_queue.size())
            tf.identity(o['index'], 'forward_output_index')
            tf.identity(o['batch_index'], 'forward_output_batch_index')
            tf.identity(o['output'], 'forward_output_output')

            # BACKWARD
            # queues
            batch_size_queue = tf.FIFOQueue(
                capacity=100,
                dtypes=[tf.int32],
                shapes=[[]])

            input_queue = tf.FIFOQueue(
                capacity=self.game_state_count * self.worker_count,
                dtypes=[tf.int32, tf.bool, tf.int64, tf.float32] + [
                    inputs[input_name]['type']
                    for input_name in inputs],
                shapes=[[], [], [self.seed_size], output['shape']] + [
                    inputs[input_name]['shape']
                    for input_name in inputs],
                names=['index', 'training', 'seed', 'output_gradient'] + [
                    input_name
                    for input_name in inputs])

            output_queue = tf.FIFOQueue(
                capacity=100,
                dtypes=[tf.int32] + [inputs[input_name]['type']
                                     for input_name in inputs
                                     if inputs[input_name]['differentiable']],
                shapes=[[]] + [inputs[input_name]['shape']
                               for input_name in inputs
                               if inputs[input_name]['differentiable']],
                names=['index'] + [input_name + '_gradient'
                                   for input_name in inputs
                                   if inputs[input_name]['differentiable']])

            # create input
            batch_size_queue.enqueue_many(
                placeholders['batch_size'], 'backward_enqueue_batch_size')

            input_queue.enqueue_many({
                'index': placeholders['index'],
                'training': placeholders['training'],
                'seed': placeholders['seed'],
                'output_gradient': placeholders['output_gradient'],
                **{
                    input_name: placeholders[input_name]
                    for input_name in inputs
                }}, 'backward_enqueue_input')

            # create flow
            batch_size = batch_size_queue.dequeue()
            batch = input_queue.dequeue_many(batch_size)

            with tf.variable_scope('transformation', reuse=True) as \
                    transformation_variable_scope:
                signal = transformation(**{
                    'training': batch['training'][0],
                    'seed': batch['seed'][0],
                    **{input_name: batch[input_name]
                       for input_name in inputs}})

            input_gradients = tf.gradients(
                signal,
                [batch[input_name]
                 for input_name in inputs
                 if inputs[input_name]['differentiable']],
                batch['output_gradient'])

            print([input_name
                   for input_name in inputs
                   if inputs[input_name]['differentiable']])
            print(input_gradients)

            self.__model_gradients(
                variable_scope, transformation_variable_scope, signal,
                batch['output_gradient'])

            with tf.control_dependencies(
                    tf.get_collection(
                        '{}/update_model_gradient_accumulators'.format(name))):
                enqueue_op = output_queue.enqueue_many({
                    'index': batch['index'],
                    **{
                        input_name + '_gradient': input_gradient
                        for input_name, input_gradient in zip(
                            [input_name
                             for input_name in inputs
                             if inputs[input_name]['differentiable']],
                            input_gradients)
                    }})
                queue_runner = tf.train.QueueRunner(output_queue, enqueue_op)
                tf.train.add_queue_runner(queue_runner)

            # create output
            outputs = output_queue.dequeue_many(output_queue.size())
            for output_name, output in outputs.items():
                tf.identity(output, 'backward_output_' + output_name)

    def __build_cost_function_graph(self, global_step):
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
                    global_step,
                    move_rates, ucb_move_rates, ugtsa_move_rates,
                    trainable_variables)

            output = tf.identity(signal, 'output')

            move_rates_gradient, = tf.gradients(output, [move_rates])
            tf.identity(move_rates_gradient, 'move_rates_gradient')

            self.__model_gradients(
                variable_scope, transformation_variable_scope, output, 1)

    def __build_apply_gradients_graph(self, global_step):
        print('apply_gradients')
        with tf.variable_scope('apply_gradients'):
            training = tf.placeholder(
                tf.bool,
                [],
                'training')

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

            self._apply_gradients(global_step, grads_and_vars)

    def build(self):
        training = tf.placeholder(tf.bool, name='training')
        global_step = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int32,
            name='global_step')

        self.__build_function_graph(
            name='empty_statistic',
            inputs={
                'game_state_board': {
                    'type': tf.float32,
                    'shape': [self.game_state_board_shape[0],
                              self.game_state_board_shape[1]],
                    'differentiable': False
                },
                'game_state_statistic': {
                    'type': tf.float32,
                    'shape': [self.game_state_statistic_size],
                    'differentiable': False
                }
            },
            output={'shape': [self.statistic_size]},
            transformation=self._empty_statistic)

        self.__build_function_graph(
            name='move_rate',
            inputs={
                'parent_statistic': {
                    'type': tf.float32,
                    'shape': [self.statistic_size],
                    'differentiable': True
                },
                'child_statistic': {
                    'type': tf.float32,
                    'shape': [self.statistic_size],
                    'differentiable': True
                }
            },
            output={'shape': [self.player_count]},
            transformation=self._move_rate)

        self.__build_function_graph(
            name='game_state_as_update',
            inputs={
                'update_statistic': {
                    'type': tf.float32,
                    'shape': [self.update_statistic_size],
                    'differentiable': False
                }
            },
            output={'shape': [self.update_size]},
            transformation=self._game_state_as_update)

        self.__build_function_graph(
            name='updated_statistic',
            inputs={
                'statistic': {
                    'type': tf.float32,
                    'shape': [self.statistic_size],
                    'differentiable': True
                },
                'update_count': {
                    'type': tf.int32,
                    'shape': [],
                    'differentiable': False
                },
                'updates': {
                    'type': tf.float32,
                    'shape': [self.update_size * self.worker_count],
                    'differentiable': True
                }
            },
            output={'shape': [self.statistic_size]},
            transformation=self._updated_statistic)

        self.__build_function_graph(
            name='updated_update',
            inputs={
                'update': {
                    'type': tf.float32,
                    'shape': [self.update_size],
                    'differentiable': True
                },
                'statistic': {
                    'type': tf.float32,
                    'shape': [self.statistic_size],
                    'differentiable': True
                }
            },
            output={'shape': [self.update_size]},
            transformation=self._updated_update)

        self.__build_cost_function_graph(global_step)
        self.__build_apply_gradients_graph(global_step)

        exit(0)

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
            trainable_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                '{}/transformation'.format(name))
            model_gradient_accumulators = tf.get_default_graph() \
                .get_collection('{}/model_gradient_accumulators'.format(name))

            print(session.run(trainable_variables[:2]))
            print(session.run(model_gradient_accumulators[:2]))

    @classmethod
    def apply_gradients(cls) -> None:
        tf.get_default_session().run(
            tf.get_default_graph().get_operation_by_name(
                'apply_gradients/apply_gradients'))
