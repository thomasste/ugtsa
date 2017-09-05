#include "ugtsa/computation_graphs/synchronized_computation_graph/computation_graph.h"

namespace computation_graphs {
namespace synchronized_computation_graph {

ComputationGraph::ComputationGraph(computation_graphs::computation_graph::ComputationGraph *computation_graph, int thread_count)
    : computation_graphs::computation_graph::ComputationGraph(), computation_graph(computation_graph), thread_count(thread_count), waiting_threads(0), barrier(thread_count) {
}

int ComputationGraph::transformation(
        std::string seed,
        int seed_size,
        std::vector<std::string> inputs,
        std::vector<std::vector<int>> input_shapes,
        std::vector<tensorflow::DataType> input_types,
        std::vector<std::string> input_gradients,
        std::string output,
        std::vector<int> output_shape,
        tensorflow::DataType output_type,
        std::string output_gradient,
        std::string update_model_gradient_accumulators) {
    return computation_graph->transformation(
        seed,
        seed_size,
        inputs,
        input_shapes,
        input_types,
        input_gradients,
        output,
        output_shape,
        output_type,
        output_gradient,
        update_model_gradient_accumulators);
}

int ComputationGraph::matrix(Eigen::VectorXi vector) {
    std::unique_lock<std::mutex> lock(mutex);
    return computation_graph->matrix(vector);
}

int ComputationGraph::matrix(Eigen::VectorXf vector) {
    std::unique_lock<std::mutex> lock(mutex);
    return computation_graph->matrix(vector);
}

int ComputationGraph::matrix(Eigen::MatrixXi matrix) {
    std::unique_lock<std::mutex> lock(mutex);
    return computation_graph->matrix(matrix);
}

int ComputationGraph::matrix(Eigen::MatrixXf matrix) {
    std::unique_lock<std::mutex> lock(mutex);
    return computation_graph->matrix(matrix);
}

int ComputationGraph::transformation_run(int transformation, std::vector<std::vector<int>> inputs) {
    std::unique_lock<std::mutex> lock(mutex);
    return computation_graph->transformation_run(transformation, inputs);
}

void ComputationGraph::run_batch() {
    {
        std::unique_lock<std::mutex> lock(mutex);

        if (waiting_threads == thread_count - 1) {
            computation_graph->run_batch();
            waiting_threads = 0;
        } else {
            waiting_threads++;
        }
    }

    barrier.wait();
}

void ComputationGraph::accumulate_model_gradients(int first_node, std::vector<int> y_grad_indices, std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> y_grad_values) {
    computation_graph->accumulate_model_gradients(first_node, y_grad_indices, y_grad_values);
}

Eigen::VectorXf ComputationGraph::value(int index) {
    std::unique_lock<std::mutex> lock(mutex);
    return computation_graph->value(index);
}

}
}