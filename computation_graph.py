from recordclass import recordclass
from itertools import chain
import numpy as np
import tensorflow as tf


class ComputationGraph:
    Function = recordclass('Function', 'name inputs input_derivatives '
                                       'output output_derivative')
    Node = recordclass('Node', 'function inputs value')
    Task = recordclass('Task', 'inputs output_derivative')

    def __init__(self, session):
        self.functions = []
        self.nodes = []
        self.batches = [0]
        self.session = session

    def function(self, inputs: [tf.Tensor], input_derivatives: [tf.Tensor],
                 output: tf.Tensor, output_derivative: tf.Tensor) -> int:
        self.functions += ComputationGraph.Function(
            inputs=inputs,
            inputs_derivatives=input_derivatives,
            output=output,
            output_derivative=output_derivative)
        return len(self.functions) - 1

    def matrix(self, matrix: np.ndarray) -> int:
        self.nodes += [ComputationGraph.Node(
            function=None,
            inputs=None,
            value=matrix
        )]
        return len(self.nodes) - 1

    def node(self, function: int, inputs: [int]) -> int:
        self.nodes += [ComputationGraph.Node(
            function=function,
            inputs=inputs,
            value=None
        )]
        return len(self.nodes) - 1

    def run_batch(self) -> None:
        self.batches += [len(self.nodes)]

        tasks = [
            ComputationGraph.Task(inputs=[[] for _ in function.inputs])
            for function in self.functions]

        for i in range(self.batches[-2], self.batches[-1]):
            node = self.nodes[i]
            if node.function is not None:
                for j, input_idx in enumerate(node.inputs):
                    tasks[node.function].inputs[j] += \
                        [self.nodes[input_idx].value]

        results = self.session.run(
            [function.output for function in self.functions],
            feed_dict={
                tensor: value
                for function, task in zip(self.functions, tasks)
                for tensor, value in zip(function.inputs, task.inputs)
            }
        )

        for i in range(self.batches[-2], self.batches[-1]):
            node = self.nodes[i]
            if node.function is not None:
                node.value, results[node.function] = \
                    results[node.function][0], results[node.function][1:]

    def calculate_derivatives(self, node: int, nodes: [int]) -> [np.ndarray]:
        derivatives = [np.zeros(node.value.shape) for node in self.nodes]
        derivatives[node] = np.ones(self.nodes[node].value.shape)

        batches = list(self.batches)

        while len(batches) > 0:
            tasks = [
                ComputationGraph.Task(
                    inputs=[[] for _ in function.inputs],
                    output_derivative=[])
                for function in self.functions]

            for i in range(batches[-2], batches[-1]):
                node = self.nodes[i]
                if node.function is not None:
                    tasks[node.function].output_derivative += [derivatives[i]]
                    for j, input_idx in enumerate(node.inputs):
                        tasks[node.function].inputs[j] += \
                            [self.nodes[input_idx].value]

            results_ = self.session.run(
                chain.from_iterable([function.input_derivatives
                                     for function in self.functions]),
                feed_dict={**{
                    tensor: value
                    for function, task in zip(self.functions, tasks)
                    for tensor, value in zip(function.inputs, task.inputs)
                }, **{
                    function.output_derivative: task.output_derivative
                    for function, task in zip(self.functions, tasks)
                }}
            )

            results = []
            for function in self.functions:
                results += results_[:len(function.inputs)]
                results_ = results_[len(function.inputs):]

            for i in range(batches[-2], batches[-1]):
                node = self.nodes[i]
                if node.function is not None:
                    for result, input_idx in zip(
                            results[node.function], node.inputs):
                        derivatives[input_idx] += result

            batches.pop()

        return [derivatives[node] for node in nodes]
