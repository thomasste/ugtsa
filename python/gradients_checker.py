from config import config
from games.game.game_state import GameState
from copy import deepcopy
from algorithms.ucb_mcts.algorithm import Algorithm as UCBAlgorithm
from computation_graphs.basic_computation_graph.computation_graph import \
    ComputationGraph
from model_builders.model_builder.model_builder import ModelBuilder

from tensorflow.python.saved_model import builder

import argparse
import logging
import numpy as np
import sys
import tensorflow as tf
import time

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('graph_path')
argument_parser.add_argument('model_path')
args = argument_parser.parse_args()

with open(args.graph_path, mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

with tf.Session() as session:
    tf.import_graph_def(graph_def, name="")


    session.run(
        tf.get_default_graph().get_operation_by_name("save/restore_all"),
        {tf.get_default_graph().get_tensor_by_name("save/Const:0"): args.model_path})
    print([n.name for n in tf.get_default_graph().as_graph_def().node if "empty_statistic/gradient_accumulator" in n.name])

    for name in [
            'empty_statistic',
            'move_rate',
            'game_state_as_update',
            'updated_statistic',
            'updated_update',
            'cost_function']:
        try:
            tensor = tf.get_default_graph().get_tensor_by_name("{}/gradient_accumulator:0".format(name))
            print(tensor)
            print(session.run(tensor))
            for i in range(1, 100):
                tensor = tf.get_default_graph().get_tensor_by_name("{}/gradient_accumulator_{}:0".format(name, i))
                print(tensor)
                print(session.run(tensor))
        except:
            pass


    # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    #
    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='empty_statistic/model_gradient_accumulators*'):
    #     print(v)
