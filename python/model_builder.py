from config import config

import argparse
import tensorflow as tf

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('game', type=str)
argument_parser.add_argument('algorithm', type=str)
argument_parser.add_argument('model_builder', type=str)
argument_parser.add_argument('worker_count', type=int)
args = argument_parser.parse_args()

game_config = config['games'][args.game]

game_state = game_config['game_state']
algorithm_config = game_config['algorithms'][args.algorithm]
algorithm_class = algorithm_config['class']
model_builder = algorithm_config['model_builders'][args.model_builder]

graph = tf.Graph()
model_builder.set_worker_count(args.worker_count)
with graph.as_default():
    model_builder.build()

with tf.Session(graph=graph) as session:
    graph_name = '{}__{}__{}__{}'.format(
        args.game, args.algorithm, args.model_builder, args.worker_count)
    model_name = '{}.0'.format(graph_name)

    # initial model
    session.run(tf.global_variables_initializer())

    # create save/restore ops
    saver = tf.train.Saver()

    # save graph
    tf.train.write_graph(
        session.graph.as_graph_def(), '../cpp/build/graphs/',
        '{}.pb'.format(graph_name),
        as_text=False)

    # save model
    saver_def = saver.as_saver_def()
    session.run(
        graph.get_operation_by_name(saver_def.save_tensor_name[:-2]),
        {graph.get_tensor_by_name(saver_def.filename_tensor_name): "../cpp/build/models/{}".format(model_name)})

    print("file_name: {}".format(saver_def.filename_tensor_name))
    print("restore_op: {}".format(saver_def.restore_op_name))
    print("save_op: {}".format(saver_def.save_tensor_name))
