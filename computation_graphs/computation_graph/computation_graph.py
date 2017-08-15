import tensorflow as tf
import numpy as np


class ComputationGraph(object):
    Node = object
    Transformation = object

    def __init__(self, training: bool):
        self.training = training

    def transformation(
            self, inputs: [tf.Tensor], input_gradients: [tf.Tensor],
            output: tf.Tensor, output_gradient: tf.Tensor,
            model_gradients: [tf.Tensor], seed: tf.Tensor,
            training: tf.Tensor) \
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

    # y_grads: Map Node [np.ndarray] -> Map Transformation [np.ndarray]
    def model_gradients(self, first_node: Node, y_grads):
        raise NotImplementedError
