import tensorflow as tf


def add_to_collection(name, tensor):
    collection = '{}/{}'.format(tf.get_variable_scope().name, name)
    print('- {}'.format(collection))
    tf.add_to_collection(collection, tensor)


def placeholder(dtype, shape=None, name=None):
    result = tf.placeholder(dtype, shape, name)
    add_to_collection(name, result)
    return result


# TODO: add untrainable model
class Model:
    def __init__(self):
        self.model = self.model_tail = \
            placeholder(tf.float32, name='model')
        self.initial_model_parts = []
        self.variables = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        initial_model = tf.concat(
            self.initial_model_parts, 0,
            name='initial_model') \
            if self.initial_model_parts \
            else tf.zeros((0, 1), name='initial_model')
        add_to_collection('initial_model', initial_model)

    def get_variable(self, initial_value, name, reuse=False):
        name = '{}/{}'.format(tf.get_variable_scope().name, name)

        if name not in self.variables:
            vectorized_initial_value = tf.reshape(initial_value, (-1, 1))
            height = vectorized_initial_value.get_shape()[0].value
            self.initial_model_parts += [vectorized_initial_value]
            vectorized_variable, self.model_tail = tf.split(
                self.model_tail, (height, -1), 0, num=2)
            self.variables[name] = tf.reshape(
                vectorized_variable, initial_value.get_shape(), name=name)
            return self.variables[name]
        elif reuse:
            return self.variables[name]
        else:
            raise Exception('reuse = False')


class ModelBuilder(object):
    def __init__(self, variable_scope, statistic_size, update_size,
                 game_state_board_shape, game_state_statistic_size,
                 update_statistic_size, player_count, worker_count):
        self.variable_scope = variable_scope
        self.statistic_size = statistic_size
        self.update_size = update_size
        self.game_state_board_shape = game_state_board_shape
        self.game_state_statistic_size = game_state_statistic_size
        self.update_statistic_size = update_statistic_size
        self.player_count = player_count
        self.worker_count = worker_count

    def _empty_statistic_transformation(
            self, model, game_state_board, game_state_statistic):
        raise NotImplementedError

    def _move_rate_transformation(
            self, model, parent_statistic, child_statistic):
        raise NotImplementedError

    def _game_state_as_update_transformation(self, model, update_statistic):
        raise NotImplementedError

    def _updated_statistic_transformation(
            self, model, statistic, update_count, updates):
        raise NotImplementedError

    def _updated_update_transformation(self, model, update, statistic):
        raise NotImplementedError

    def __build_empty_statistic_graph(self):
        print('empty_statistic')
        with tf.variable_scope('empty_statistic'):
            with Model() as model:
                game_state_board = placeholder(
                    tf.float32,
                    [None,
                     self.game_state_board_shape[0],
                     self.game_state_board_shape[1]],
                    name='game_state_board')
                game_state_statistic = placeholder(
                    tf.float32,
                    [None, self.game_state_statistic_size],
                    name='game_state_statistic')

                with tf.variable_scope('transformation'):
                    signal = self._empty_statistic_transformation(
                        model, game_state_board, game_state_statistic)

                output = tf.identity(signal, name='output')
                output_gradient = placeholder(
                    tf.float32, [None, self.statistic_size],
                    name='output_gradient')
                model_gradient, game_state_board_gradient, \
                    game_state_statistic = tf.gradients(
                        output,
                        [model.model, game_state_board, game_state_statistic],
                        grad_ys=output_gradient)

                for name in ['output',
                             'model_gradient',
                             'game_state_board_gradient',
                             'game_state_statistic']:
                    add_to_collection(name, locals()[name])

    def __build_move_rate_graph(self):
        print('move_rate')
        with tf.variable_scope('move_rate'):
            with Model() as model:
                parent_statistic = placeholder(
                    tf.float32,
                    [None, self.statistic_size],
                    name='parent_statistic')
                child_statistic = placeholder(
                    tf.float32,
                    [None, self.statistic_size],
                    name='child_statistic')

                with tf.variable_scope('transformation'):
                    signal = self._move_rate_transformation(
                        model, parent_statistic, child_statistic)

                output = tf.identity(signal, name='output')
                output_gradient = placeholder(
                    tf.float32,
                    [None, self.player_count],
                    name='output_gradient')
                model_gradient, parent_statistic_gradient, \
                    child_statistic_gradient = tf.gradients(
                        output, [model.model, parent_statistic,
                                 child_statistic], grad_ys=output_gradient)

                for name in ['output',
                             'model_gradient',
                             'parent_statistic_gradient',
                             'child_statistic_gradient']:
                    add_to_collection(name, locals()[name])

    def __build_game_state_as_update_graph(self):
        print('game_state_as_update')
        with tf.variable_scope('game_state_as_update'):
            with Model() as model:
                update_statistic = placeholder(
                    tf.float32,
                    [None, self.update_statistic_size],
                    name='update_statistic')

                with tf.variable_scope('transformation'):
                    signal = self._game_state_as_update_transformation(
                        model, update_statistic)

                output = tf.identity(signal, name='output')
                output_gradient = placeholder(
                    tf.float32,
                    [None, self.update_size],
                    name='output_gradient')
                model_gradient, update_statistic_gradient = tf.gradients(
                    output, [model.model, update_statistic], grad_ys=output_gradient)

                for name in ['output',
                             'model_gradient',
                             'update_statistic_gradient']:
                    add_to_collection(name, locals()[name])

    def __build_updated_statistic_graph(self):
        print('updated_statistic')
        with tf.variable_scope('updated_statistic'):
            with Model() as model:
                statistic = placeholder(
                    tf.float32,
                    [None, self.statistic_size],
                    name='statistic')
                update_count = placeholder(
                    tf.int32,
                    [None],
                    name='update_count')
                updates = placeholder(
                    tf.float32,
                    [None, self.update_size * self.worker_count],
                    name='updates')

                with tf.variable_scope('transformation'):
                    signal = self._updated_statistic_transformation(
                        model, statistic, update_count, updates)

                output = tf.identity(signal, name='output')
                output_gradient = placeholder(
                    tf.float32,
                    [None, self.statistic_size],
                    name='output_gradient')
                model_gradient, statistic_gradient, update_count, \
                    update_gradient = tf.gradients(
                        output,
                        [model.model, statistic, update_count, updates],
                        grad_ys=output_gradient)

                for name in ['output',
                             'model_gradient',
                             'statistic_gradient',
                             'update_count',
                             'update_gradient']:
                    add_to_collection(name, locals()[name])

    def __build_updated_update_graph(self):
        print('updated_update')
        with tf.variable_scope('updated_update'):
            with Model() as model:
                update = placeholder(
                    tf.float32,
                    [None, self.update_size],
                    name='update')
                statistic = placeholder(
                    tf.float32,
                    [None, self.statistic_size],
                    name='statistic')

                with tf.variable_scope('transformation'):
                    signal = self._updated_update_transformation(
                        model, update, statistic)

                output = tf.identity(signal, name='output')
                output_gradient = placeholder(
                    tf.float32,
                    [None, self.update_size],
                    name='output_gradient')
                model_gradient, statistic_gradient, \
                    update_gradient = tf.gradients(
                        output, [model.model, statistic, update],
                        grad_ys=output_gradient)

                for name in ['output',
                             'model_gradient',
                             'statistic_gradient',
                             'update_gradient']:
                    add_to_collection(name, locals()[name])

    def build(self):
        with tf.variable_scope(self.variable_scope):
            with tf.variable_scope('settings'):
                self.training = placeholder(tf.bool, name='training')

            self.__build_empty_statistic_graph()
            self.__build_move_rate_graph()
            self.__build_game_state_as_update_graph()
            self.__build_updated_statistic_graph()
            self.__build_updated_update_graph()