from computation_graphs.computation_graph import computation_graph
from recordclass import recordclass
from threading import RLock, Semaphore

import tensorflow as tf
import numpy as np
import threading


class ComputationGraph(computation_graph.ComputationGraph):
    Node = object
    Transformation = object

    Thread = recordclass('Thread', 'semaphore is_waiting')

    def __init__(self, computation_graph: computation_graph.ComputationGraph):
        self.computation_graph = computation_graph

        self.lock = RLock()
        self.threads = {}

    def transformation(
            self, inputs: [tf.Tensor], input_gradients: [tf.Tensor],
            output: tf.Tensor, output_gradient: tf.Tensor,
            model_gradients: [tf.Tensor], seed: tf.Tensor) \
            -> Transformation:
        with self.lock:
            return self.computation_graph.transformation(
                inputs, input_gradients, output, output_gradient,
                model_gradients, seed)

    def matrix(self, matrix: np.ndarray) -> Node:
        with self.lock:
            return self.computation_graph.matrix(matrix)

    # inputs: Union [Node] [[Node]]
    def transformation_run(self, transformation: Node, inputs) -> Node:
        with self.lock:
            return self.computation_graph.transformation_run(
                transformation, inputs)

    def __waiting_threads_count(self):
        counter = 0
        for thread in self.threads.values():
            if thread.is_waiting:
                counter += 1
        return counter

    def run_batch(self) -> None:
        semaphore = None

        with self.lock:
            if self.__waiting_threads_count() == len(self.threads) - 1:
                self.computation_graph.run_batch()
                for thread in self.threads.values():
                    if thread.is_waiting:
                        thread.semaphore.release()
            else:
                thread = self.threads[threading.get_ident()]
                thread.is_waiting = True
                semaphore = thread.semaphore

        if semaphore is not None:
            semaphore.acquire()

    def value(self, node_index: Node) -> np.ndarray:
        with self.lock:
            return self.computation_graph.value(node_index)

    # : Map Transformation [np.ndarray]
    def model_gradients(self, first_node: Node, y_grads: [(Node, np.ndarray)]):
        with self.lock:
            return self.model_gradients(first_node, y_grads)

    def add_thread(self) -> None:
        with self.lock:
            self.threads[threading.get_ident()] = ComputationGraph.Thread(
                semaphore=Semaphore(value=0), is_waiting=False)

    def remove_thread(self) -> None:
        with self.lock:
            self.threads.pop(threading.get_ident())
            if self.__waiting_threads_count() == len(self.threads):
                self.computation_graph.run_batch()
                for thread in self.threads.values():
                    if thread.is_waiting:
                        thread.semaphore.release()
