from recordclass import recordclass

import numpy as np
import tensorflow as tf


class ComputationGraph:
    Transformation = recordclass(
        'Transformation', 'batch_inputs batch_input_gradients '
                          'inputs input_gradients input_shapes '
                          'output output_gradient output_shape')
    # nodes are not topologically sorted inside batches
    Node = recordclass('Node', 'transformation inputs output')
    ForwardTask = recordclass('ForwardTask', 'batch_inputs inputs')
    BackwardTask = recordclass(
        'BackwardTask', 'batch_inputs inputs output_gradient')

    def __init__(self, session: tf.Session):
        self.session = session
        self.transformations = []
        self.nodes = []
        self.batches = [0]
        self.batch_inputs = []

    def __tensorflow_shape_to_numpy_shape(self, shape):
        return [x if x is not None else -1 for x in shape.as_list()]

    def transformation(
            self, batch_inputs: [tf.Tensor],
            batch_input_gradients: [tf.Tensor], inputs: [tf.Tensor],
            input_gradients: [tf.Tensor], output: tf.Tensor,
            output_gradient: tf.Tensor) -> int:
        self.transformations += [
            ComputationGraph.Transformation(
                batch_inputs=batch_inputs,
                batch_input_gradients=batch_input_gradients,
                inputs=inputs,
                input_gradients=input_gradients,
                input_shapes=[
                    self.__tensorflow_shape_to_numpy_shape(input.get_shape())
                    for input in inputs],
                output=output,
                output_gradient=output_gradient,
                output_shape=self.__tensorflow_shape_to_numpy_shape(
                    output.get_shape()))]
        return len(self.transformations) - 1

    def matrix(self, matrix: np.ndarray) -> int:
        self.nodes += [
            ComputationGraph.Node(
                transformation=None,
                inputs=None,
                output=matrix)]
        return len(self.nodes) - 1

    # inputs: [Either int [int]]
    def transformation_run(
            self, transformation: int, inputs) -> int:
        self.nodes += [
            ComputationGraph.Node(
                transformation=transformation,
                inputs=inputs,
                output=None)]
        return len(self.nodes) - 1

    def __prepared_input(self, input, input_shape):
        if type(input) is list:
            concatenated_input = np.concatenate(
                [self.nodes[node_idx].output for node_idx in input])
            expanded_input = np.lib.pad(
                concatenated_input,
                (0, input_shape[-1] - concatenated_input.shape[0]),
                'constant')
            return expanded_input
        else:
            return self.nodes[input].output

    # batch_inputs: Map int [int]
    def run_batch(self, batch_inputs) -> None:
        self.batches += [len(self.nodes)]
        self.batch_inputs += [batch_inputs]

        tasks = [
            ComputationGraph.ForwardTask(
                batch_inputs=[],
                inputs=[[] for _ in transformation.inputs])
            for transformation in self.transformations]

        for transformation_index, bis in batch_inputs.items():
            task = tasks[transformation_index]
            for batch_input in bis:
                task.batch_inputs += [self.nodes[batch_input].output]

        for i in range(self.batches[-2], self.batches[-1]):
            node = self.nodes[i]
            if node.transformation is not None:
                transformation = self.transformations[node.transformation]
                task = tasks[node.transformation]
                for task_input, node_input, input_shape in zip(
                        task.inputs, node.inputs,
                        transformation.input_shapes):
                    task_input += [self.__prepared_input(
                        node_input, input_shape)]

        fetches = [
            transformation.output
            for transformation in self.transformations]
        batch_input_feed_dict = {
            k: v
            for transformation, task in zip(self.transformations, tasks)
            for k, v in zip(transformation.batch_inputs, task.batch_inputs)}
        input_feed_dict = {
            k: np.reshape(v, shape)
            for transformation, task in zip(self.transformations, tasks)
            for k, v, shape in zip(
                transformation.inputs, task.inputs,
                transformation.input_shapes)}

        results = self.session.run(
            fetches=fetches,
            feed_dict={
                **batch_input_feed_dict,
                **input_feed_dict})

        for i in range(self.batches[-2], self.batches[-1]):
            node = self.nodes[i]
            if node.transformation is not None:
                node.output, results[node.transformation] = \
                    results[node.transformation][0], \
                    results[node.transformation][1:]

    def gradients(self, xs: [int], y: int, y_gradient=None):
        gradients = [
            np.zeros(node.output.shape, dtype=node.output.dtype)
            for node in self.nodes]

        if y_gradient is not None:
            gradients[y] = y_gradient
        else:
            gradients[y] = np.ones(
                self.nodes[y].output.shape, dtype=self.nodes[y].output.dtype)

        batches = list(self.batches)
        batch_inputs = list(self.batch_inputs)

        while len(batches) > 1:
            tasks = [
                ComputationGraph.BackwardTask(
                    batch_inputs=[],
                    inputs=[[] for _ in transformation.inputs],
                    output_gradient=[])
                for transformation in self.transformations]

            for transformation_index, bis in batch_inputs[-1].items():
                task = tasks[transformation_index]
                for batch_input in bis:
                    task.batch_inputs += [self.nodes[batch_input].output]

            for i in range(batches[-2], batches[-1]):
                node = self.nodes[i]
                if node.transformation is not None:
                    transformation = self.transformations[node.transformation]
                    task = tasks[node.transformation]

                    task.output_gradient += [gradients[i]]

                    for task_input, node_input, input_shape in zip(
                            task.inputs, node.inputs,
                            transformation.input_shapes):
                        task_input += [self.__prepared_input(
                            node_input, input_shape)]

            batch_input_gradient_fetches = [
                transformation.batch_input_gradients
                for transformation in self.transformations]
            input_gradient_fetches = [
                transformation.input_gradients
                for transformation in self.transformations]
            batch_input_feed_dict = {
                k: v
                for transformation, task in zip(self.transformations, tasks)
                for k, v in zip(
                    transformation.batch_inputs, task.batch_inputs)}
            input_feed_dict = {
                k: np.reshape(v, shape)
                for transformation, task in zip(self.transformations, tasks)
                for k, v, shape in zip(
                    transformation.inputs, task.inputs,
                    transformation.input_shapes)}
            output_gradient_feed_dict = {
                transformation.output_gradient: np.reshape(
                    task.output_gradient, transformation.output_shape)
                for transformation, task in zip(self.transformations, tasks)}

            results = self.session.run(
                fetches={
                    'batch_input_gradient': batch_input_gradient_fetches,
                    'input_gradient_fetches': input_gradient_fetches,
                },
                feed_dict={
                    **batch_input_feed_dict,
                    **input_feed_dict,
                    **output_gradient_feed_dict})

            for transformation_index, task in enumerate(tasks):
                for result, batch_input in zip(
                        results['batch_input_gradient'][transformation_index],
                        batch_inputs[-1][transformation_index]):
                    gradients[batch_input] += result

            for i in range(batches[-2], batches[-1]):
                node = self.nodes[i]
                if node.transformation is not None:
                    transformation_results = \
                        results['input_gradient_fetches'][node.transformation]
                    for input_index, node_input in enumerate(node.inputs):
                        input_result = transformation_results[input_index][0]
                        transformation_results[input_index] = \
                            transformation_results[input_index][1:]
                        if type(node_input) is list:
                            for ni in node_input:
                                gradients[ni] += input_result[
                                    :self.nodes[ni].output.shape[0]]
                                input_result = input_result[
                                    self.nodes[ni].output.shape[0]:]
                        else:
                            gradients[node_input] += input_result

            batches.pop()
            batch_inputs.pop()

        return [gradients[x] for x in xs]

    def value(self, node: int) -> np.ndarray:
        return self.nodes[node].output
