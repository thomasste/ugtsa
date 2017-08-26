import tensorflow as tf
import numpy as np


class ComputationGraph(object):
    Node = object
    Transformation = object

    def __init__(self, training: bool):
        self.training = training

    def transformation(self, training: tf.Tensor, seed: tf.Tensor,
                       inputs: [(tf.Tensor, tf.Tensor)],
                       output: (tf.Tensor, tf.Tensor),
                       update_model_gradient_accumulators: [tf.Operation])\
            -> Transformation:
        raise NotImplementedError

    def matrix(self, matrix: np.ndarray) -> Node:
        raise NotImplementedError

    # inputs: Union [Node] [[Node]]
    def transformation_run(self, transformation: Node, inputs) -> Node:
        raise NotImplementedError

    def run_batch(self) -> None:
        raise NotImplementedError

    def value(self, node_index: Node) -> np.ndarray:
        raise NotImplementedError

    # y_grads: Map Node [np.ndarray]
    def model_gradients(self, first_node: Node, y_grads) -> None:
        raise NotImplementedError
