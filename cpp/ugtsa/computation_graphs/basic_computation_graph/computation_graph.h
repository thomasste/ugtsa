#pragma once

#include "ugtsa/computation_graphs/computation_graph/computation_graph.h"

namespace computation_graphs {
namespace basic_computation_graph {

class ComputationGraph : public computation_graphs::computation_graph::ComputationGraph {
    struct Transformation {
        std::string training;
        std::string seed;
        int seed_size;
        std::vector<std::string> inputs;
        std::vector<std::vector<int>> input_shapes;
        std::vector<int> input_sizes;
        std::vector<tensorflow::DataTypes> input_types;
        std::vector<std::string> input_gradients;
        std::string output;
        std::vector<int> ouptut_shape;
        int output_size;
        std::vector<std::string> output_gradient;
    };

    struct Node {
        int transformation;
        std::vector<std::vector<int>> inputs;
        std::vector<int> output;
    };

    struct Batch {
        int nodes_end;
        std::vector<std::vector<long long>> seeds;
    };

    tensorflow::Session* session;

    std::default_random_engine generator;
    std::vector<Transformation> transformations;
    std::vector<Nodes> nodes;
    std::vector<Batch> batches;

public:
    ComputationGraph(bool training, tensorflow::Session* session);

    int transformation(
        std::string training,
        std::string seed,
        int seed_size,
        std::vector<std::string> inputs,
        std::vector<std::vector<int>> input_shapes,
        std::vector<tensorflow::DataTypes> input_types,
        std::vector<std::string> input_gradients,
        std::string output,
        std::vector<int> ouptut_shape,
        std::vector<std::string> output_gradient);
    int matrix(Eigen::VectorXi vector);
    int matrix(Eigen::VectorXf vector);
    int matrix(Eigen::MatrixXi matrix);
    int matrix(Eigen::MatrixXf matrix);
    int transformation_run(int transformation, std::vector<std::vector<int>> inputs);
    void run_batch();
    Eigen::VectorXf value(int index);
};

}
}