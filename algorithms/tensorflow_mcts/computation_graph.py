from recordclass import recordclass
import numpy as np
import tensorflow as tf


class ComputationGraph:
    Transformation = recordclass(
        'Transformation', 'model model_gradient '
                          'inputs input_gradients '
                          'output output_gradient '
                          'model_matrix '
                          'input_lengths')
    Node = recordclass('Node', 'transformation inputs value')
    ForwardTask = recordclass('ForwardTask', 'inputs')

    def __init__(self, session: tf.Session):
        self.session = session
        self.transformations = []
        self.nodes = []
        self.batches = [0]

    def transformation(self,
            model: [tf.Tensor], model_gradient: [tf.Tensor],
            inputs: [tf.Tensor], input_gradients: [tf.Tensor],
            output: tf.Tensor, output_gradient: tf.Tensor,
            model_matrix: int) -> int:
        # print([input.get_shape()
        #        for input in inputs])
        self.transformations += [
            ComputationGraph.Transformation(
                model=model,
                model_gradient=model_gradient,
                inputs=inputs,
                input_lengths=[
                    input.get_shape().as_list()[1] if len(input.get_shape().as_list()) > 1 else None
                    for input in inputs],
                input_gradients=input_gradients,
                output=output,
                output_gradient=output_gradient,
                model_matrix=model_matrix)]
        return len(self.transformations) - 1

    def matrix(self, matrix: np.ndarray) -> int:
        self.nodes += [
            ComputationGraph.Node(
                transformation=None,
                inputs=None,
                value=matrix)]
        return len(self.nodes) - 1

    # def node(self, transformation: int, inputs: Union[[int], [[int]]]) \
    #         -> int:
    def node(self, transformation: int, inputs) \
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
                    # print(type(input))
                    if type(input) is list:
                        tmp_1 = np.concatenate(
                            [self.nodes[node_idx].value for node_idx in input])
                        # print(tmp_1)
                        # print(input_length)
                        # print(type(input_length))
                        # print(tmp_1.shape[0])
                        # print(type(tmp_1.shape[0]))
                        # print((0, input_length - tmp_1.shape[0]))
                        tmp_2 = np.lib.pad(
                            tmp_1, (0, input_length - tmp_1.shape[0]),
                            'constant')
                        task_input += [tmp_2]
                    else:
                        task_input += [self.nodes[input].value]

        # print("dupa")
        # print("dupa")
        # print("dupa")
        # print("dupa")
        # print("dupa")
        # print("dupa")

        # print(self.nodes)
        # print(self.transformations)
        # print(tasks)

        # print({
        #     **{
        #         tensor: np.reshape(value, [x if x is not None else -1 for x in tensor.get_shape().as_list()])
        #         for transformation, task in zip(
        #             self.transformations, tasks)
        #         for tensor, value in zip(
        #             transformation.inputs, task.inputs)
        #     },
        #     **{
        #         transformation.model:
        #             self.nodes[transformation.model_matrix].value
        #         for transformation in self.transformations
        #     }
        # })

        results = self.session.run(
            [transformation.output
             for task, transformation in zip(tasks, self.transformations)],
            feed_dict={
                **{
                    tensor: np.reshape(value, [x if x is not None else -1 for x in tensor.get_shape().as_list()])
                    for transformation, task in zip(
                        self.transformations, tasks)
                    for tensor, value in zip(
                        transformation.inputs, task.inputs)
                },
                **{
                    transformation.model:
                        self.nodes[transformation.model_matrix].value
                    for transformation in self.transformations
                }
            })

        # print(results)
        # print(results[0][0][0])
        # print(type(results[0][0][0]))
        # exit(0)

        for i in range(self.batches[-2], self.batches[-1]):
            node = self.nodes[i]
            if node.transformation is not None:
                node.value, results[node.transformation][0] = \
                    results[node.transformation][0][0], \
                    results[node.transformation][0][1:]

    def value(self, node: int) -> np.ndarray:
        return self.nodes[node].value
