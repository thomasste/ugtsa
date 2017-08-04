from recordclass import recordclass
from typing import Union
import numpy as np
import tensorflow as tf


class ComputationGraph:
    Transformation = recordclass(
        'Transformation', 'trainable_model trainable_model_gradient '
                          'inputs input_gradients '
                          'output output_gradient '
                          'trainable_model_matrix '
                          'input_lengths')
    Node = recordclass('Node', 'transformation inputs value')
    ForwardTask = recordclass('ForwardTask', 'inputs')

    def __init__(self, session: tf.Session):
        self.session = session
        self.transformations = []
        self.nodes = []
        self.batches = [0]

    def transformation(self,
            trainable_model: [tf.Tensor], trainable_model_gradient: [tf.Tensor],
            inputs: [tf.Tensor], input_gradients: [tf.Tensor],
            output: tf.Tensor, output_gradient: tf.Tensor,
            trainable_model_matrix: int) -> int:
        self.transformations += [
            ComputationGraph.Transformation(
                trainable_model=trainable_model,
                trainable_model_gradient=trainable_model_gradient,
                inputs=inputs,
                input_lengths=[input.get_shape()[1] for input in inputs],
                input_gradients=input_gradients,
                output=output,
                output_gradient=output_gradient,
                trainable_model_matrix=trainable_model_matrix)]
        return len(self.transformations) - 1

    def matrix(self, matrix: np.ndarray) -> int:
        self.nodes += [
            ComputationGraph.Node(
                transformation=None,
                inputs=None,
                value=matrix)]
        return len(self.nodes) - 1

    def node(self, transformation: int, inputs: Union[[int], [[int]]]) \
            -> int:
        self.nodes += [
            ComputationGraph.Node(
                transformation=transformation,
                inputs=inputs,
                value=None)]
        return len(self.nodes) - 1

    def run_batch(self) -> None:
        self.batches += [len(self.nodes)]

        tasks = [
            ComputationGraph.ForwardTask(
                inputs=[[] for _ in transformation.inputs])
            for transformation in self.transformations]

        for i in range(self.batches[-2], self.batches[-1]):
            node = self.nodes[i]
            if node.transformation is not None:
                transformation = self.transformations[node.transformation]
                for input, input_length, task_input in zip(
                        node.inputs, transformation.input_lengths,
                        tasks[node.transformation].inputs):
                    if type(input) is list:
                        tmp_1 = np.concatenate(
                            [self.nodes[node_idx] for node_idx in input])
                        tmp_2 = np.lib.pad(
                            tmp_1, (0, input_length - tmp_1.shape[0]),
                            'constant')
                        task_input += [tmp_2]
                    else:
                        task_input += [self.nodes[input]]

        results = self.session.run(
            [transformation.output
             for transformation in self.transformations],
            feed_dict={
                **{
                    tensor: value
                    for transformation, task in zip(
                        self.transformations, tasks)
                    for tensor, value in zip(
                        transformation.inputs, task.inputs)
                },
                **{
                    transformation.trainable_model:
                        self.nodes[transformation.trainable_model_matrix].value
                    for transformation in self.transformations
                }
            })

        for i in range(self.batches[-2], self.batches[-1]):
            node = self.nodes[i]
            if node.transformation is not None:
                node.value, results[node.transformation] = \
                    results[node.transformation][0], \
                    results[node.transformation][1:]

    def value(self, node: int) -> np.ndarray:
        return self.nodes[node].value
